/**
 * State reset for the benchmark.
 *
 * A run has to start from the same state as the previous one, or the numbers it
 * reports depend on how many times the tool has already been run against this
 * server. The cart is an in-process singleton, so this just calls the store's
 * reset.
 *
 * Linked from nowhere in the UI, for the same reason as the manifest: an
 * exploring agent must not be able to find it.
 */

import { NextResponse } from "next/server";

import { reset } from "@/lib/store";

export const dynamic = "force-dynamic";

export function POST() {
  return NextResponse.json({ ok: true, cart: reset() });
}
