import { NextResponse } from "next/server";

import { type SortKey, queryCatalog } from "@/lib/products";

export const dynamic = "force-dynamic";

export function GET(request: Request) {
  const url = new URL(request.url);
  const sort = (url.searchParams.get("sort") as SortKey | null) ?? "name";
  const page = Number.parseInt(url.searchParams.get("page") ?? "1", 10) || 1;
  const perPage = Number.parseInt(url.searchParams.get("perPage") ?? "6", 10) || 6;

  const result = queryCatalog({
    q: url.searchParams.get("q") ?? undefined,
    sort,
    page,
    perPage,
  });

  return NextResponse.json(result);
}
