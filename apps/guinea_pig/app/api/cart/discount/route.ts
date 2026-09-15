import { NextResponse } from "next/server";

import { applyDiscount } from "@/lib/store";

export const dynamic = "force-dynamic";

export async function POST(request: Request) {
  let code: unknown;
  try {
    ({ code } = (await request.json()) as { code?: unknown });
  } catch {
    return NextResponse.json({ ok: false, error: "Malformed JSON body." }, { status: 400 });
  }

  if (typeof code !== "string" || code.trim() === "") {
    return NextResponse.json({ ok: false, error: "code is required." }, { status: 400 });
  }

  const result = applyDiscount(code);
  return NextResponse.json(result, { status: result.ok ? 200 : 400 });
}
