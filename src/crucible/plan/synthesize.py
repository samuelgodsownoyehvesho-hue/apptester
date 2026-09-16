"""Test case synthesis from the app map.

The planner selects executable checks and ranks them by risk. It does not
write prose test cases: a case is only worth generating if it can run, and the
registry in :mod:`crucible.execute.checks` is the source of what *can* run.

Risk ranking is deliberately simple and explainable — lane weights, plus a
boost when the app map shows the surface is money-handling. A rubric nobody
can audit is worse than a simple one everybody can.
"""

from __future__ import annotations

from dataclasses import dataclass

from crucible.recon.scout import AppMapData

#: Base weight per lane. Money arithmetic outranks everything because a wrong
#: total is a direct business loss; accessibility matters but rarely blocks.
LANE_RISK = {
    "arithmetic": 4,
    "metamorphic": 3,
    "http": 3,
    "accessibility": 2,
    "ui": 2,
}


@dataclass(frozen=True, slots=True)
class PlannedCase:
    """One selected check, with its traceability reference and risk."""

    check_id: str
    title: str
    lane: str
    risk: int
    #: Traceability anchor. Here it is the check id itself, which the runner
    #: resolves against the registry. A case whose reference resolves to
    #: nothing is unexecutable and must be visible as such.
    requirement_ref: str


def synthesize_cases(app_map: AppMapData | None) -> list[PlannedCase]:
    """Select checks that apply to the discovered surface.

    The current registry is target-shaped (a shop with a cart and a catalog),
    so every check is selected whenever its surface was observed, and the
    cart/catalog probes are selected unconditionally: failing to probe because
    recon missed a link would trade recall for tidiness. Risk is raised when
    the app map confirms the surface exists.
    """
    observed_paths: set[str] = set(app_map.route_paths) if app_map else set()
    has_api = bool(app_map and (app_map.api_operations or _saw_api_paths(observed_paths)))

    planned: list[PlannedCase] = []

    def add(check_id: str, title: str, lane: str, *, surface_seen: bool = True) -> None:
        risk = LANE_RISK.get(lane, 2) + (1 if surface_seen else 0)
        planned.append(
            PlannedCase(
                check_id=check_id,
                title=title,
                lane=lane,
                risk=risk,
                requirement_ref=check_id,
            )
        )

    add(
        "cart_quantity_arithmetic",
        "Cart total equals the sum of unit price times quantity",
        "arithmetic",
    )
    add(
        "rounding_precision",
        "A discounted total rounds to the nearest cent instead of being cut off",
        "arithmetic",
    )
    add("empty_cart_reset", "Emptying the cart resets the total to zero", "metamorphic")
    add("discount_idempotence", "Applying one discount code twice changes nothing", "metamorphic")
    add(
        "price_sort_monotonic",
        "Price-ascending sort returns non-decreasing prices",
        "metamorphic",
        surface_seen="/catalog" in observed_paths or app_map is None,
    )
    add(
        "search_case_equivalence",
        "Search results are identical regardless of query casing",
        "metamorphic",
        surface_seen="/catalog" in observed_paths or app_map is None,
    )
    add(
        "pagination_disjoint",
        "Consecutive catalog pages share no items",
        "metamorphic",
        surface_seen="/catalog" in observed_paths or app_map is None,
    )
    add(
        "negative_quantity_rejected",
        "Cart API rejects a quantity below 1",
        "http",
        surface_seen=has_api or app_map is None,
    )
    add(
        "referenced_routes_respond",
        "Every page route linked from the layout responds",
        "http",
        # This check probes the routes recon found, so its usefulness tracks
        # discovered paths rather than the API surface.
        surface_seen=bool(observed_paths) or app_map is None,
    )

    # Highest risk first: if a budget stop fires mid-run, the expensive
    # lessons have already been learned.
    return sorted(planned, key=lambda case: (-case.risk, case.check_id))


def _saw_api_paths(paths: set[str]) -> bool:
    return any(path.startswith("/api") for path in paths)
