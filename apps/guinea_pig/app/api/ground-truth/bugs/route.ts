/**
 * Ground-truth manifest for the benchmark.
 *
 * Linked from nowhere in the UI. A browser agent exploring the application
 * should never discover it — if it did, it could read the answers instead of
 * finding them, and every recall number afterwards would be meaningless.
 *
 * This route used to live at /api/_benchmark/bugs. In the App Router a folder
 * prefixed with `_` is *private* and excluded from routing, so that path
 * returned 404 and the harness could never read the manifest at all. Do not
 * reintroduce an underscore segment: the protection here is that nothing links
 * to this route, not that its name is hard to guess.
 *
 * The benchmark harness calls this directly to learn which defects are active
 * in this instance.
 */

import { NextResponse } from "next/server";

import { BUG_CATALOG, BUG_IDS, activeBugDefinitions } from "@/lib/bugs";

export const dynamic = "force-dynamic";

export function GET() {
  return NextResponse.json({
    schemaVersion: 1,
    selection: process.env.GUINEA_PIG_BUGS ?? "all",
    allIds: BUG_IDS,
    activeIds: activeBugDefinitions().map((bug) => bug.id),
    catalog: BUG_CATALOG,
  });
}
