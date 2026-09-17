"""Executable checks.

A check gathers **facts only**: what was requested, what came back, and which
invariant probe was applied. Whether a fact constitutes a defect is decided
later by the oracle. The split is what allows the oracle to be re-run over
stored results, and what keeps a flaky transport from being reported as a bug.

Every check is a small state machine over the target's public surface:
arrange (seed state), act (probe), report. They never assert.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from crucible.core.logging import get_logger
from crucible.core.qa import ChatChannel
from crucible.execute.client import AppClient, reset_target
from crucible.recon.scout import AppMapData

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class CheckResult:
    """What one check observed."""

    check_id: str
    lane: str
    #: The invariant this probe was designed to exercise, phrased as a fact
    #: about what happened rather than a claim about correctness.
    observation: str
    #: Structured facts the oracle's signals consume. Kept deliberately rich:
    #: the oracle sees only this, never the live server.
    facts: dict[str, Any] = field(default_factory=dict)
    #: Transport or arrangement failure. A check that could not run has
    #: observed nothing and must not contribute evidence either way.
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "check_id": self.check_id,
            "lane": self.lane,
            "observation": self.observation,
            "facts": self.facts,
            "error": self.error,
        }


# ---------------------------------------------------------------- cart state
# The guinea pig keeps cart state in a single in-process store, so probes run
# in sequence against the same session. Each check is responsible for leaving
# the cart empty afterwards where it can, but the runner tolerates residue.


async def _empty_cart(client: AppClient) -> None:
    """Return the cart to a known state before a check probes it.

    Prefers the target's own reset because clearing the lines is not enough:
    every check shares one cart, so a discount left behind by the rounding or
    discount probe changes the figures the next probe reports. The invariant
    still detects the defect either way, but the numbers in the report stop
    making sense to anyone reading them.
    """
    if await reset_target(client):
        return

    snapshot = await client.get("/api/cart")
    lines: list[dict[str, Any]] = []
    if snapshot.is_json and isinstance(snapshot.json_body, dict):
        lines = snapshot.json_body.get("lines") or []
    for line in lines:
        product_id = line.get("productId")
        if isinstance(product_id, str):
            await client.delete("/api/cart", params={"productId": product_id})


# ------------------------------------------------------------------- checks


async def check_cart_quantity_arithmetic(client: AppClient, questioner: ChatChannel | None = None) -> CheckResult:
    """Add 2 units of a $10.00 item and record the reported total."""
    check_id = "cart_quantity_arithmetic"
    await _empty_cart(client)

    added = await client.post(
        "/api/cart",
        {"productId": "probe-1", "name": "Probe Item", "price": 10.00, "quantity": 2},
    )
    if added.error or not added.is_json:
        return CheckResult(check_id, "arithmetic", "cart add failed", {}, added.error)

    cart = added.json_body.get("cart", added.json_body) if isinstance(added.json_body, dict) else {}
    await _empty_cart(client)

    return CheckResult(
        check_id,
        "arithmetic",
        "added 2 x $10.00 and read back the total",
        {
            "quantity": 2,
            "unit_price": 10.00,
            "expected_total": 20.00,
            "reported_total": cart.get("total"),
            "reported_subtotal": cart.get("subtotal"),
            "lines": cart.get("lines", []),
        },
    )


async def check_rounding_precision(client: AppClient, questioner: ChatChannel | None = None) -> CheckResult:
    """Apply a discount that leaves a fraction of a cent on the total.

    $1.07 less 20% is $0.85600. Rounding to the nearest cent gives $0.86;
    dropping the third digit gives $0.85. The two disagree, which is the only
    way to tell which one the application does. A probe without a discount
    produces an exact result either way and can never answer the question --
    which is why the previous attempt at this check could only stay silent.
    """
    check_id = "rounding_precision"
    await _empty_cart(client)

    added = await client.post(
        "/api/cart",
        {"productId": "probe-4", "name": "Probe Item", "price": 1.07, "quantity": 1},
    )
    if added.error or not added.is_json:
        return CheckResult(check_id, "arithmetic", "cart add failed", {}, added.error)

    discounted = await client.post("/api/cart/discount", {"code": "SAVE20"})
    if discounted.error or not discounted.is_json:
        return CheckResult(check_id, "arithmetic", "discount failed", {}, discounted.error)

    body = discounted.json_body if isinstance(discounted.json_body, dict) else {}
    cart = body.get("cart", body)
    await _empty_cart(client)

    return CheckResult(
        check_id,
        "arithmetic",
        "applied a discount leaving a fraction of a cent",
        {
            "invariant": (
                "a total is rounded to the nearest cent, not cut off at the "
                "third decimal"
            ),
            "unit_price": 1.07,
            "discount_code": "SAVE20",
            # The expectation is derived from what the server *reported*, not
            # from what we asked for, so a wrong subtotal elsewhere cannot be
            # mistaken for a rounding defect.
            "subtotal": cart.get("subtotal"),
            "discount_rate": cart.get("discountRate"),
            "reported_total": cart.get("total"),
        },
    )


async def check_empty_cart_reset(client: AppClient, questioner: ChatChannel | None = None) -> CheckResult:
    """Add an item, remove it, and read the total again."""
    check_id = "empty_cart_reset"
    await _empty_cart(client)

    await client.post(
        "/api/cart",
        {"productId": "probe-2", "name": "Probe Item", "price": 10.00, "quantity": 1},
    )
    removed = await client.delete("/api/cart", params={"productId": "probe-2"})
    if removed.error or not removed.is_json:
        return CheckResult(check_id, "metamorphic", "cart remove failed", {}, removed.error)

    final = await client.get("/api/cart")
    if final.error or not final.is_json:
        return CheckResult(check_id, "metamorphic", "cart readback failed", {}, final.error)

    body = final.json_body if isinstance(final.json_body, dict) else {}
    return CheckResult(
        check_id,
        "metamorphic",
        "emptied the cart and read back the total",
        {
            "invariant": "cart with zero items has total == 0",
            "line_count": len(body.get("lines", [])),
            "reported_total": body.get("total"),
            "reported_subtotal": body.get("subtotal"),
        },
    )


async def check_discount_idempotence(client: AppClient, questioner: ChatChannel | None = None) -> CheckResult:
    """Apply SAVE10 twice; idempotence means the rate stays 0.1."""
    check_id = "discount_idempotence"
    await _empty_cart(client)

    first = await client.post("/api/cart/discount", {"code": "SAVE10"})
    if first.error or not first.is_json:
        return CheckResult(check_id, "metamorphic", "first discount failed", {}, first.error)
    # The discount endpoint answers with an envelope ({"ok": ..., "cart": ...})
    # rather than a bare cart, so the rates live one level down. Reading them
    # off the envelope yields None for both, which the signal reports as
    # "evidence unusable" -- a silent miss rather than a false positive.
    first_body = first.json_body if isinstance(first.json_body, dict) else {}
    first_cart = first_body.get("cart", first_body)
    first_rate = first_cart.get("discountRate")
    first_total = first_cart.get("total")

    second = await client.post("/api/cart/discount", {"code": "SAVE10"})
    if second.error or not second.is_json:
        return CheckResult(check_id, "metamorphic", "second discount failed", {}, second.error)
    second_body = second.json_body if isinstance(second.json_body, dict) else {}
    second_cart = second_body.get("cart", second_body)

    await _empty_cart(client)

    return CheckResult(
        check_id,
        "metamorphic",
        "applied the same discount code twice",
        {
            "invariant": "applying a discount twice equals applying it once (idempotence)",
            "code": "SAVE10",
            "first_rate": first_rate,
            "second_rate": second_cart.get("discountRate"),
            "first_total": first_total,
            "second_total": second_cart.get("total"),
        },
    )


async def check_price_sort_monotonic(client: AppClient, questioner: ChatChannel | None = None) -> CheckResult:
    """Request price_asc over a wide page and record the price sequence."""
    check_id = "price_sort_monotonic"
    listing = await client.get("/api/products", params={"sort": "price_asc", "perPage": "50"})
    if listing.error or not listing.is_json:
        return CheckResult(check_id, "metamorphic", "catalog listing failed", {}, listing.error)

    body = listing.json_body if isinstance(listing.json_body, dict) else {}
    items = body.get("items", [])
    prices = [item.get("price") for item in items if isinstance(item, dict)]
    names = [item.get("name") for item in items if isinstance(item, dict)]

    return CheckResult(
        check_id,
        "metamorphic",
        "requested the catalog sorted by price ascending",
        {
            "invariant": "results sorted by price are non-decreasing numerically",
            "sort": "price_asc",
            "prices_in_order": prices,
            "names_in_order": names,
            "total": body.get("total"),
        },
    )


async def check_search_case_equivalence(client: AppClient, questioner: ChatChannel | None = None) -> CheckResult:
    """Search the same term in two casings; record both result sets."""
    check_id = "search_case_equivalence"
    lower = await client.get("/api/products", params={"q": "laptop", "perPage": "50"})
    upper = await client.get("/api/products", params={"q": "LAPTOP", "perPage": "50"})
    if lower.error or upper.error:
        return CheckResult(
            check_id, "metamorphic", "catalog search failed", {},
            lower.error or upper.error,
        )

    def items_of(response: object) -> list[dict[str, Any]]:
        if isinstance(response, dict):
            items = response.get("items", [])
            return [i for i in items if isinstance(i, dict)]
        return []

    def ids(response: object) -> list[str]:
        return [str(item.get("id")) for item in items_of(response)]

    # Product names are captured alongside the ids purely so a reader gets
    # "laptop stand" rather than "p-03" in the report.
    def names(response: object) -> list[str]:
        return [str(item.get("name")) for item in items_of(response)]

    return CheckResult(
        check_id,
        "metamorphic",
        'searched "laptop" and "LAPTOP"',
        {
            "invariant": 'search("laptop") returns the same set as search("LAPTOP")',
            "lower_ids": ids(lower.json_body),
            "upper_ids": ids(upper.json_body),
            "lower_names": names(lower.json_body),
            "upper_names": names(upper.json_body),
        },
    )


async def check_pagination_disjoint(client: AppClient, questioner: ChatChannel | None = None) -> CheckResult:
    """Fetch pages 1 and 2 and record whether any product id appears on both."""
    check_id = "pagination_disjoint"
    page1 = await client.get("/api/products", params={"page": "1", "perPage": "6"})
    page2 = await client.get("/api/products", params={"page": "2", "perPage": "6"})
    if page1.error or page2.error:
        return CheckResult(
            check_id, "metamorphic", "catalog pagination failed", {},
            page1.error or page2.error,
        )

    def ids(response: object) -> list[str]:
        if isinstance(response, dict):
            items = response.get("items", [])
            return [str(i.get("id")) for i in items if isinstance(i, dict)]
        return []

    ids1, ids2 = ids(page1.json_body), ids(page2.json_body)
    overlap = sorted(set(ids1) & set(ids2))

    return CheckResult(
        check_id,
        "metamorphic",
        "fetched catalog pages 1 and 2",
        {
            "invariant": "consecutive pages do not repeat an item",
            "page1_ids": ids1,
            "page2_ids": ids2,
            "overlap": overlap,
        },
    )


async def check_negative_quantity_rejected(client: AppClient, questioner: ChatChannel | None = None) -> CheckResult:
    """Post quantity -1 and record the status the server returned."""
    check_id = "negative_quantity_rejected"
    response = await client.post(
        "/api/cart",
        {"productId": "probe-3", "name": "Probe Item", "price": 10.00, "quantity": -1},
    )
    if response.error:
        return CheckResult(check_id, "http", "cart add request failed", {}, response.error)

    await _empty_cart(client)

    return CheckResult(
        check_id,
        "http",
        "posted a cart line with quantity -1",
        {
            "invariant": "POST /api/cart rejects quantity < 1 with 400",
            "status": response.status,
            "body": response.json_body if response.is_json else response.text[:500],
        },
    )


async def check_referenced_routes_respond(
    client: AppClient,
    app_map: AppMapData | None = None,
    questioner: ChatChannel | None = None,
) -> CheckResult:
    """Probe the routes recon actually discovered and record each status.

    The route list comes from the app map, never from a literal. A hardcoded
    list silently asserts that the target has *our* pages: against any other
    application every probe 404s and the oracle reports broken links that were
    never there -- a false positive manufactured by the harness itself.

    When a route redirects to a login page, the bot pauses and asks the human
    operator whether to skip behind the wall or provide credentials.
    """
    check_id = "referenced_routes_respond"
    routes = sorted(app_map.route_paths) if app_map is not None else []
    if not routes:
        return CheckResult(
            check_id,
            "http",
            "recon discovered no routes to probe",
            {"invariant": "every internal link returns a non-error status"},
        )

    observed: dict[str, int] = {}
    login_walls: list[str] = []
    skipped: list[str] = []

    for route in routes:
        response = await client.get(route)
        status = response.status if not response.error else 0

        # Detect login redirects: the server sends the bot to /login or a
        # similar path instead of returning the page.
        is_login_redirect = (
            status in (301, 302, 303, 307, 308)
            and isinstance(response.text, str)
            and any(kw in response.text.lower() for kw in ("/login", "/signin", "/auth"))
        )

        if is_login_redirect and questioner is not None and route not in login_walls:
            login_walls.append(route)

    # Ask the human about login walls in one batch rather than per-route.
    skip_login = False
    credentials: dict[str, str] | None = None
    if login_walls and questioner is not None:
        wall_list = ", ".join(login_walls)
        answer = await questioner.ask(
            f"I found pages that require login: {wall_list}",
            options=[
                "Skip those pages",
                "I'll provide login credentials",
            ],
            context=(
                "These routes redirect to a login page instead of showing "
                "their content. I can skip them, or you can give me a "
                "username and password to log in."
            ),
        )
        answer_lower = answer.strip().lower()
        if "skip" in answer_lower:
            skip_login = True
            await questioner.inform("OK — skipping pages that require login.")
        elif "credential" in answer_lower or "provide" in answer_lower:
            creds_answer = await questioner.ask(
                "Please provide credentials as: username password",
                context="For example: admin mysecretpass",
            )
            parts = creds_answer.strip().split(None, 1)
            if len(parts) == 2:
                credentials = {"username": parts[0], "password": parts[1]}
                await questioner.inform(
                    f"Got it — will try to log in as {credentials['username']}."
                )
            else:
                await questioner.inform(
                    "Couldn't parse credentials. Skipping login pages."
                )
                skip_login = True

    # Now actually probe each route.
    for route in routes:
        response = await client.get(route)
        status = response.status if not response.error else 0

        is_login_redirect = (
            status in (301, 302, 303, 307, 308)
            and isinstance(response.text, str)
            and any(kw in response.text.lower() for kw in ("/login", "/signin", "/auth"))
        )

        if is_login_redirect:
            if skip_login:
                skipped.append(route)
                continue
            if credentials is not None:
                # Try to authenticate: POST to the login endpoint.
                login_response = await client.post(
                    "/login",
                    {"username": credentials["username"], "password": credentials["password"]},
                )
                if not login_response.error and login_response.status in (200, 302):
                    # Retry the original route after login.
                    response = await client.get(route)
                    status = response.status if not response.error else 0

        observed[route] = status

    detail_parts = [f"requested the {len(routes)} route(s) recon discovered"]
    if skipped:
        detail_parts.append(f"skipped {len(skipped)} route(s) behind login walls")
    if credentials:
        detail_parts.append(f"authenticated as {credentials['username']}")

    return CheckResult(
        check_id,
        "http",
        ". ".join(detail_parts),
        {
            "invariant": "every internal link returns a non-error status",
            "statuses": observed,
            "login_walls": login_walls,
            "skipped_routes": skipped,
        },
    )


#: The registry the plan stage selects from. Ordered so the cheapest probes
#: run first and shared-state probes (which mutate the cart) run before ones
#: that depend on a clean cart.
CHECK_REGISTRY: dict[str, Any] = {
    "cart_quantity_arithmetic": check_cart_quantity_arithmetic,
    "rounding_precision": check_rounding_precision,
    "empty_cart_reset": check_empty_cart_reset,
    "discount_idempotence": check_discount_idempotence,
    "price_sort_monotonic": check_price_sort_monotonic,
    "search_case_equivalence": check_search_case_equivalence,
    "pagination_disjoint": check_pagination_disjoint,
    "negative_quantity_rejected": check_negative_quantity_rejected,
}

#: Checks that need recon's findings in order to probe honestly. Kept separate
#: from :data:`CHECK_REGISTRY` so the extra argument is explicit at the call
#: site rather than every check taking a context it never uses.
CONTEXT_CHECK_REGISTRY: dict[str, Any] = {
    "referenced_routes_respond": check_referenced_routes_respond,
}
