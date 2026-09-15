import { NextResponse } from "next/server";

import { addLine, getCart, removeLine } from "@/lib/store";

export const dynamic = "force-dynamic";

export function GET() {
  return NextResponse.json(getCart());
}

interface AddPayload {
  productId?: unknown;
  name?: unknown;
  price?: unknown;
  quantity?: unknown;
}

export async function POST(request: Request) {
  let payload: AddPayload;
  try {
    payload = (await request.json()) as AddPayload;
  } catch {
    return NextResponse.json({ ok: false, error: "Malformed JSON body." }, { status: 400 });
  }

  const { productId, name, price } = payload;
  const quantity = payload.quantity ?? 1;

  if (
    typeof productId !== "string" ||
    typeof name !== "string" ||
    typeof price !== "number" ||
    typeof quantity !== "number"
  ) {
    return NextResponse.json(
      { ok: false, error: "productId, name, price and quantity are required." },
      { status: 400 },
    );
  }

  const result = addLine(productId, name, price, quantity);

  if (!result.ok) {
    // A rejected quantity must surface as 400, not a 200 wrapping an error,
    // otherwise an HTTP-level check cannot distinguish the two.
    return NextResponse.json(result, { status: 400 });
  }

  return NextResponse.json(result, { status: 201 });
}

export function DELETE(request: Request) {
  const productId = new URL(request.url).searchParams.get("productId");
  if (!productId) {
    return NextResponse.json({ ok: false, error: "productId is required." }, { status: 400 });
  }
  return NextResponse.json(removeLine(productId));
}
