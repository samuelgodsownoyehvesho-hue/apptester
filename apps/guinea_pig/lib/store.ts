/**
 * Cart state and pricing.
 *
 * A module-level singleton, so state survives across requests for the lifetime
 * of the dev server and resets on restart. That is deliberate: it makes the
 * target cheap to reset between runs without a database.
 *
 * Every defect here is guarded by a `bugOn` check so the file reads as the
 * correct implementation plus explicitly labelled faults.
 */

import { bugOn } from "./bugs";

export interface CartLine {
  productId: string;
  name: string;
  unitPrice: number;
  quantity: number;
}

export interface CartSnapshot {
  lines: CartLine[];
  appliedCodes: string[];
  discountRate: number;
  subtotal: number;
  discount: number;
  total: number;
}

const DISCOUNT_CODES: Readonly<Record<string, number>> = {
  SAVE10: 0.1,
  SAVE20: 0.2,
  HALFOFF: 0.5,
};

interface CartState {
  lines: CartLine[];
  appliedCodes: string[];
  discountRate: number;
  subtotal: number;
  total: number;
}

const state: CartState = {
  lines: [],
  appliedCodes: [],
  discountRate: 0,
  subtotal: 0,
  total: 0,
};

function round2(value: number): number {
  return Math.round(value * 100) / 100;
}

function computeSubtotal(lines: CartLine[]): number {
  if (bugOn("CART_QTY_IGNORED")) {
    // Defect: quantity is not multiplied in. A line of 3 x $19.99 contributes
    // $19.99 rather than $59.97.
    return round2(lines.reduce((sum, line) => sum + line.unitPrice, 0));
  }
  return round2(
    lines.reduce((sum, line) => sum + line.unitPrice * line.quantity, 0),
  );
}

function computeTotal(subtotal: number, discountRate: number): number {
  const raw = subtotal * (1 - discountRate);
  if (bugOn("TRUNCATED_ROUNDING")) {
    // Defect: truncation rather than rounding, so money is lost whenever the
    // third decimal is 5 or more.
    return Math.trunc(raw * 100) / 100;
  }
  return round2(raw);
}

/** Recompute derived fields from the current lines and discount. */
function recalc(): void {
  state.subtotal = computeSubtotal(state.lines);
  state.total = computeTotal(state.subtotal, state.discountRate);
}

function snapshot(): CartSnapshot {
  return {
    lines: state.lines.map((line) => ({ ...line })),
    appliedCodes: [...state.appliedCodes],
    discountRate: state.discountRate,
    subtotal: state.subtotal,
    discount: round2(state.subtotal * state.discountRate),
    total: state.total,
  };
}

export function getCart(): CartSnapshot {
  return snapshot();
}

export interface AddResult {
  ok: boolean;
  error?: string;
  cart: CartSnapshot;
}

/**
 * Add an item to the cart.
 *
 * Exported separately from the HTTP layer so both the browser and the API
 * exercise the same code path — a defect has to be reachable from the UI to be
 * a realistic browser-testing target.
 */
export function addLine(
  productId: string,
  name: string,
  unitPrice: number,
  quantity: number,
): AddResult {
  if (!bugOn("CHECKOUT_ACCEPTS_NEGATIVE_QTY")) {
    if (!Number.isInteger(quantity) || quantity < 1) {
      return {
        ok: false,
        error: "Quantity must be a whole number of at least 1.",
        cart: snapshot(),
      };
    }
  }

  const existing = state.lines.find((line) => line.productId === productId);
  if (existing) {
    existing.quantity += quantity;
  } else {
    state.lines.push({ productId, name, unitPrice, quantity });
  }

  recalc();
  return { ok: true, cart: snapshot() };
}

export interface RemoveResult {
  ok: boolean;
  error?: string;
  cart: CartSnapshot;
}

/** Remove every line for a product. */
export function removeLine(productId: string): RemoveResult {
  state.lines = state.lines.filter((line) => line.productId !== productId);

  if (bugOn("EMPTY_CART_STALE_TOTAL") && state.lines.length === 0) {
    // Defect: returning before recalculating leaves the previous total on
    // screen once the cart is emptied.
    return { ok: true, cart: snapshot() };
  }

  recalc();
  return { ok: true, cart: snapshot() };
}

export interface DiscountResult {
  ok: boolean;
  error?: string;
  cart: CartSnapshot;
}

/** Apply a discount code. */
export function applyDiscount(code: string): DiscountResult {
  const normalized = code.trim().toUpperCase();
  const rate = DISCOUNT_CODES[normalized];

  if (rate === undefined) {
    return { ok: false, error: `Unknown discount code "${code}".`, cart: snapshot() };
  }

  if (bugOn("DISCOUNT_STACKS")) {
    // Defect: codes compound rather than being capped, so applying the same
    // code twice yields 19% off instead of 10%.
    state.discountRate = 1 - (1 - state.discountRate) * (1 - rate);
  } else {
    state.discountRate = Math.max(state.discountRate, rate);
  }

  if (!state.appliedCodes.includes(normalized)) {
    state.appliedCodes.push(normalized);
  }

  recalc();
  return { ok: true, cart: snapshot() };
}

/** Empty the cart completely. Used by the benchmark between runs. */
export function reset(): CartSnapshot {
  state.lines = [];
  state.appliedCodes = [];
  state.discountRate = 0;
  state.subtotal = 0;
  state.total = 0;
  return snapshot();
}
