/**
 * Mutant registry.
 *
 * This application is a test target, so its defects are declared rather than
 * accidental. Each entry carries the metadata the benchmark needs to score a
 * run: where the bug lives, how severe it is, and — critically — which oracle
 * signal is *supposed* to catch it.
 *
 * Toggling
 * --------
 * `GUINEA_PIG_BUGS` selects which defects are active, so the same codebase can
 * produce many mutants without edits:
 *
 *   GUINEA_PIG_BUGS=all          every defect (the default)
 *   GUINEA_PIG_BUGS=none         a clean baseline, for measuring false positives
 *   GUINEA_PIG_BUGS=CART_QTY,CASE_SENSITIVE_SEARCH
 *
 * A run against `none` that still reports bugs is measuring the agent's
 * precision, which matters as much as recall and is much easier to get wrong.
 */

export type Severity = "low" | "medium" | "high";

/** Which verification strategy should be able to detect this defect. */
export type Detection = "spec" | "metamorphic" | "arithmetic" | "accessibility" | "http";

export interface BugDefinition {
  readonly id: string;
  readonly title: string;
  readonly area: string;
  readonly severity: Severity;
  readonly detection: Detection;
  /** The invariant that is violated. This is the oracle's ground truth. */
  readonly invariant: string;
  /** Where a white-box triage agent should end up looking. */
  readonly location: string;
}

export const BUG_CATALOG: Readonly<Record<string, BugDefinition>> = {
  CART_QTY_IGNORED: {
    id: "CART_QTY_IGNORED",
    title: "Cart total ignores item quantity",
    area: "cart",
    severity: "high",
    detection: "arithmetic",
    invariant: "cartTotal == sum(unitPrice * quantity) for every line item",
    location: "apps/guinea_pig/lib/store.ts",
  },
  EMPTY_CART_STALE_TOTAL: {
    id: "EMPTY_CART_STALE_TOTAL",
    title: "Cart total is stale after the last item is removed",
    area: "cart",
    severity: "high",
    detection: "metamorphic",
    invariant: "cart with zero items has total == 0",
    location: "apps/guinea_pig/lib/store.ts",
  },
  DISCOUNT_STACKS: {
    id: "DISCOUNT_STACKS",
    title: "Discount code can be applied more than once",
    area: "cart",
    severity: "high",
    detection: "metamorphic",
    invariant: "applying a discount twice equals applying it once (idempotence)",
    location: "apps/guinea_pig/lib/store.ts",
  },
  TRUNCATED_ROUNDING: {
    id: "TRUNCATED_ROUNDING",
    title: "Currency totals truncate instead of rounding to two decimals",
    area: "cart",
    severity: "medium",
    detection: "arithmetic",
    invariant: "displayed total equals the exact sum of line items to 2 decimals",
    location: "apps/guinea_pig/lib/store.ts",
  },
  LEXICOGRAPHIC_SORT: {
    id: "LEXICOGRAPHIC_SORT",
    title: '"Price: low to high" sorts prices as strings',
    area: "catalog",
    severity: "medium",
    detection: "metamorphic",
    invariant: "results sorted by price are non-decreasing numerically",
    location: "apps/guinea_pig/app/catalog/page.tsx",
  },
  PAGINATION_OVERLAP: {
    id: "PAGINATION_OVERLAP",
    title: "Consecutive catalog pages repeat an item",
    area: "catalog",
    severity: "medium",
    detection: "metamorphic",
    invariant: "concatenating all pages reproduces the full set exactly once",
    location: "apps/guinea_pig/app/catalog/page.tsx",
  },
  CASE_SENSITIVE_SEARCH: {
    id: "CASE_SENSITIVE_SEARCH",
    title: "Search is case sensitive",
    area: "catalog",
    severity: "low",
    detection: "metamorphic",
    invariant: 'search("laptop") returns the same set as search("LAPTOP")',
    location: "apps/guinea_pig/app/catalog/page.tsx",
  },
  UNLABELED_CHECKOUT_INPUT: {
    id: "UNLABELED_CHECKOUT_INPUT",
    title: "Checkout email field has no associated label",
    area: "checkout",
    severity: "medium",
    detection: "accessibility",
    invariant: "every form control has an accessible name",
    location: "apps/guinea_pig/app/checkout/page.tsx",
  },
  CHECKOUT_ACCEPTS_NEGATIVE_QTY: {
    id: "CHECKOUT_ACCEPTS_NEGATIVE_QTY",
    title: "Checkout API accepts a negative quantity",
    area: "api",
    severity: "high",
    detection: "http",
    invariant: "POST /api/cart rejects quantity < 1 with 400",
    location: "apps/guinea_pig/app/api/cart/route.ts",
  },
  BROKEN_FOOTER_LINK: {
    id: "BROKEN_FOOTER_LINK",
    title: "Footer links to a route that does not exist",
    area: "navigation",
    severity: "low",
    detection: "http",
    invariant: "every internal link returns a non-error status",
    location: "apps/guinea_pig/app/layout.tsx",
  },
};

export const BUG_IDS = Object.freeze(Object.keys(BUG_CATALOG));

function parseSelection(): Set<string> {
  const raw = (process.env.GUINEA_PIG_BUGS ?? "all").trim();

  if (raw === "" || raw.toLowerCase() === "all") {
    return new Set(BUG_IDS);
  }
  if (raw.toLowerCase() === "none") {
    return new Set();
  }

  const requested = raw
    .split(",")
    .map((token) => token.trim().toUpperCase())
    .filter(Boolean);

  const unknown = requested.filter((token) => !(token in BUG_CATALOG));
  if (unknown.length > 0) {
    // Failing loudly matters: a typo would otherwise silently produce fewer
    // bugs than expected and quietly deflate the recall measurement.
    throw new Error(
      `Unknown GUINEA_PIG_BUGS entries: ${unknown.join(", ")}. ` +
        `Known ids: ${BUG_IDS.join(", ")}`,
    );
  }

  return new Set(requested);
}

let cached: Set<string> | null = null;

/** Return the set of currently active defect ids. */
export function activeBugs(): ReadonlySet<string> {
  if (cached === null) {
    cached = parseSelection();
  }
  return cached;
}

/** Whether a specific defect is currently active. */
export function bugOn(id: string): boolean {
  return activeBugs().has(id);
}

/** Ground-truth definitions for the active defects only. */
export function activeBugDefinitions(): BugDefinition[] {
  return BUG_IDS.filter(bugOn).map((id) => BUG_CATALOG[id]);
}
