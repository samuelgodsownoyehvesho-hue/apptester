import { NextResponse } from "next/server";

import { getCart } from "@/lib/store";

export const dynamic = "force-dynamic";

/** Deliberately simple: a real address validator is not the interesting part. */
const EMAIL_PATTERN = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

const CARD_PATTERN = /^\d{16}$/;

export async function POST(request: Request) {
  let email: unknown;
  let card: unknown;
  try {
    ({ email, card } = (await request.json()) as { email?: unknown; card?: unknown });
  } catch {
    return NextResponse.json({ ok: false, error: "Malformed JSON body." }, { status: 400 });
  }

  if (typeof email !== "string" || !EMAIL_PATTERN.test(email.trim())) {
    return NextResponse.json({ ok: false, error: "A valid email address is required." }, { status: 400 });
  }

  const digits = typeof card === "string" ? card.replace(/\s+/g, "") : "";
  if (!CARD_PATTERN.test(digits)) {
    return NextResponse.json({ ok: false, error: "Card number must be 16 digits." }, { status: 400 });
  }

  const cart = getCart();
  if (cart.lines.length === 0) {
    return NextResponse.json({ ok: false, error: "Your cart is empty." }, { status: 400 });
  }

  const orderId = `NS-${String(Math.floor(Math.random() * 9000) + 1000)}`;

  return NextResponse.json({ ok: true, orderId, total: cart.total }, { status: 201 });
}
