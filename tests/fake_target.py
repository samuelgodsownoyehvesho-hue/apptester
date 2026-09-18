"""An in-memory simulation of the guinea pig shop.

The pipeline tests need a target that behaves like the real Next.js app —
including its bugs — without a dev server. This fake implements the same
endpoints with the same defect semantics, keyed on a ``simulate`` set so a
test can choose which mutants are active. It is a *simulation*, not a copy:
only the observable HTTP behaviour is reproduced, which is exactly what the
oracle is allowed to see.
"""

from __future__ import annotations

import math
from typing import Any

from crucible.execute.client import ApiResponse

#: Mirrors the app's DISCOUNT_CODES, keyed by the code the caller sends.
DISCOUNT_CODES: dict[str, float] = {"SAVE10": 0.1, "SAVE20": 0.2, "HALFOFF": 0.5}


def _round2(value: float) -> float:
    """Half-up to two decimals, matching JavaScript's Math.round.

    Python's built-in round() is banker's rounding, which would disagree with
    the application on exact halves and make the twin an unfaithful oracle.
    """
    return math.floor(value * 100 + 0.5) / 100


class FakeGuineaPig:
    """Minimal behavioural twin of the demo app's HTTP surface."""

    def __init__(self, *, simulate: set[str] | None = None, with_manifest: bool = True) -> None:
        self.simulate = simulate or set()
        self.with_manifest = with_manifest
        self.lines: list[dict[str, Any]] = []
        self.discount_rate = 0.0
        self._closed = False
        #: EMPTY_CART_STALE_TOTAL: totals the app failed to recompute after the
        #: last line was removed, which then stick in subsequent read-backs.
        self.stale_totals: dict[str, Any] | None = None

    # -- helpers -------------------------------------------------------

    def _ensure_open(self) -> None:
        """Fail loudly when used after close, exactly as a real transport does.

        httpx raises rather than quietly failing, and a double that kept
        answering hid a genuine bug: the pipeline closed the client and then ran
        every check through it, so each one errored, the oracle declined to call
        that evidence, and the run reported a single finding while looking
        entirely healthy. A fake that cannot fail cannot catch that.
        """
        if self._closed:
            raise RuntimeError("Cannot send a request, as the client has been closed.")

    def _fresh_cart(self) -> dict[str, Any]:
        subtotal = _round2(
            sum(float(line["price"]) * int(line["quantity"]) for line in self.lines)
        )
        if "CART_QTY_IGNORED" in self.simulate:
            subtotal = _round2(sum(float(line["price"]) for line in self.lines))
        raw = subtotal * (1 - self.discount_rate)
        truncate = "TRUNCATED_ROUNDING" in self.simulate
        total = math.trunc(raw * 100) / 100 if truncate else _round2(raw)
        return {
            "lines": [dict(line) for line in self.lines],
            "appliedCodes": ["SAVE10"] if self.discount_rate else [],
            "discountRate": self.discount_rate,
            "subtotal": subtotal,
            "discount": _round2(subtotal * self.discount_rate),
            "total": total,
        }

    def _cart(self) -> dict[str, Any]:
        """The cart the server reports, defects included.

        Under EMPTY_CART_STALE_TOTAL the stale totals are part of *stored*
        state, not just the immediate response: an app that forgets to
        recompute keeps reporting them on later reads, which is what makes the
        defect observable at all.
        """
        cart = self._fresh_cart()
        if self.stale_totals is not None and not self.lines:
            cart.update(self.stale_totals)
        return cart

    @staticmethod
    def _json(body: Any, status: int = 200) -> ApiResponse:
        return ApiResponse(status=status, is_json=True, json_body=body)

    @staticmethod
    def _products() -> list[dict[str, Any]]:
        return [
            {"id": "p-01", "name": "Laptop Pro 14", "price": 149.99},
            {"id": "p-02", "name": "Laptop Air 13", "price": 89.99},
            {"id": "p-03", "name": "laptop stand", "price": 19.99},
            {"id": "p-04", "name": "Mechanical Keyboard", "price": 129.5},
            {"id": "p-05", "name": "Wireless Mouse", "price": 9.99},
            {"id": "p-06", "name": "27-inch Monitor", "price": 319.0},
            {"id": "p-07", "name": "Ultrawide Monitor", "price": 549.95},
            {"id": "p-08", "name": "USB-C Dock", "price": 79.0},
        ]

    # -- AppClient protocol ---------------------------------------------

    async def get(self, path: str, *, params: dict[str, str] | None = None) -> ApiResponse:
        self._ensure_open()
        params = params or {}
        if path == "/api/cart":
            return self._json(self._cart())
        if path == "/api/products":
            return self._json(self._catalog(params))
        if path == "/api/ground-truth/bugs":
            if not self.with_manifest:
                return ApiResponse(status=404, text="not found")
            return self._json(
                {
                    "selection": ",".join(sorted(self.simulate)) or "none",
                    "allIds": [
                        "CART_QTY_IGNORED",
                        "EMPTY_CART_STALE_TOTAL",
                        "DISCOUNT_STACKS",
                        "TRUNCATED_ROUNDING",
                        "LEXICOGRAPHIC_SORT",
                        "PAGINATION_OVERLAP",
                        "CASE_SENSITIVE_SEARCH",
                        "UNLABELED_CHECKOUT_INPUT",
                        "CHECKOUT_ACCEPTS_NEGATIVE_QTY",
                        "BROKEN_FOOTER_LINK",
                    ],
                    "activeIds": sorted(self.simulate),
                }
            )
        if path in {"/", "/catalog", "/cart", "/checkout", "/orders"}:
            return ApiResponse(status=200, text="<html></html>")
        if path == "/returns":
            status = 404 if "BROKEN_FOOTER_LINK" in self.simulate else 200
            return ApiResponse(status=status, text="<html></html>")
        return ApiResponse(status=404, text="not found")

    async def post(self, path: str, json_body: Any) -> ApiResponse:
        self._ensure_open()
        if path == "/api/ground-truth/reset":
            self.lines = []
            self.discount_rate = 0.0
            self.stale_totals = None
            return self._json({"ok": True, "cart": self._cart()})
        if path == "/api/cart":
            quantity = json_body.get("quantity", 1)
            if "CHECKOUT_ACCEPTS_NEGATIVE_QTY" not in self.simulate and (
                not isinstance(quantity, int) or quantity < 1
            ):
                return self._json({"ok": False, "error": "bad quantity"}, 400)
            self.lines.append(
                {
                    "productId": json_body["productId"],
                    "name": json_body["name"],
                    "price": json_body["price"],
                    "quantity": quantity,
                }
            )
            # Adding a line forces a recompute, clearing any stale totals.
            self.stale_totals = None
            return self._json({"ok": True, "cart": self._cart()}, 201)
        if path == "/api/cart/discount":
            code = str(json_body.get("code", "")).upper()
            rate = DISCOUNT_CODES.get(code)
            if rate is None:
                return self._json({"ok": False, "error": "unknown code"}, 400)
            if "DISCOUNT_STACKS" in self.simulate:
                self.discount_rate = round(
                    1 - (1 - self.discount_rate) * (1 - rate), 6
                )
            else:
                self.discount_rate = max(self.discount_rate, rate)
            self.stale_totals = None
            return self._json({"ok": True, "cart": self._cart()})
        return ApiResponse(status=404, text="not found")

    async def delete(self, path: str, *, params: dict[str, str] | None = None) -> ApiResponse:
        self._ensure_open()
        if path == "/api/cart":
            product_id = (params or {}).get("productId", "")
            before = self._fresh_cart()
            self.lines = [line for line in self.lines if line["productId"] != product_id]
            if "EMPTY_CART_STALE_TOTAL" in self.simulate and not self.lines:
                # Deliberately stale: the totals are never recomputed once the
                # cart empties, so they survive into later reads.
                self.stale_totals = {
                    "subtotal": before["subtotal"],
                    "discount": before["discount"],
                    "total": before["total"],
                }
            else:
                self.stale_totals = None
            return self._json({"ok": True, "cart": self._cart()})
        return ApiResponse(status=404, text="not found")

    async def aclose(self) -> None:
        self._closed = True

    async def __aenter__(self) -> FakeGuineaPig:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.aclose()

    # -- catalog --------------------------------------------------------

    def _catalog(self, params: dict[str, str]) -> dict[str, Any]:
        items = self._products()
        query = params.get("q")
        if query is not None and query != "":
            if "CASE_SENSITIVE_SEARCH" in self.simulate:
                items = [item for item in items if query in item["name"]]
            else:
                needle = query.lower()
                items = [item for item in items if needle in item["name"].lower()]

        sort = params.get("sort", "name")
        if sort == "price_asc":
            if "LEXICOGRAPHIC_SORT" in self.simulate:
                items = sorted(items, key=lambda item: str(item["price"]))
            else:
                items = sorted(items, key=lambda item: float(item["price"]))
        elif sort == "price_desc":
            items = sorted(items, key=lambda item: -float(item["price"]))
        else:
            items = sorted(items, key=lambda item: str(item["name"]))

        per_page = int(params.get("perPage", "6"))
        try:
            page = max(1, int(params.get("page", "1")))
        except ValueError:
            page = 1
        offset = (page - 1) * per_page
        if "PAGINATION_OVERLAP" in self.simulate:
            shifted = max(0, offset - 1)
            window = items[shifted : shifted + per_page]
        else:
            window = items[offset : offset + per_page]

        return {
            "items": window,
            "total": len(items),
            "page": page,
            "perPage": per_page,
            "pages": max(1, -(-len(items) // per_page)),
        }
