import Link from "next/link";

import { AddToCartButton } from "@/app/_components/AddToCartButton";
import { DEFAULT_PER_PAGE, type SortKey, queryCatalog } from "@/lib/products";

interface SearchParams {
  q?: string;
  sort?: string;
  page?: string;
}

const SORT_OPTIONS: { value: SortKey; label: string }[] = [
  { value: "name", label: "Name (A-Z)" },
  { value: "price_asc", label: "Price: low to high" },
  { value: "price_desc", label: "Price: high to low" },
];

function buildHref(params: { q?: string; sort?: string; page?: number }): string {
  const search = new URLSearchParams();
  if (params.q) search.set("q", params.q);
  if (params.sort) search.set("sort", params.sort);
  if (params.page && params.page > 1) search.set("page", String(params.page));
  const query = search.toString();
  return query ? `/catalog?${query}` : "/catalog";
}

export default async function CatalogPage({
  searchParams,
}: {
  // Next 15 delivers searchParams as a promise.
  searchParams: Promise<SearchParams>;
}) {
  const params = await searchParams;
  const q = params.q ?? "";
  const sort = (params.sort as SortKey | undefined) ?? "name";
  const page = Number.parseInt(params.page ?? "1", 10) || 1;

  const result = queryCatalog({ q, sort, page, perPage: DEFAULT_PER_PAGE });

  return (
    <>
      <h1>Catalog</h1>
      <p className="muted">
        {result.total} product{result.total === 1 ? "" : "s"}
        {q ? ` matching “${q}”` : ""}
      </p>

      <form className="toolbar" method="get" action="/catalog">
        <label htmlFor="q">Search</label>
        <input id="q" name="q" defaultValue={q} placeholder="Product name" />
        <label htmlFor="sort">Sort</label>
        <select id="sort" name="sort" defaultValue={sort}>
          {SORT_OPTIONS.map((option) => (
            <option key={option.value} value={option.value}>
              {option.label}
            </option>
          ))}
        </select>
        <button type="submit">Apply</button>
      </form>

      {result.items.length === 0 ? (
        <p className="muted" data-testid="empty-catalog">
          No products matched that search.
        </p>
      ) : (
        <div className="grid">
          {result.items.map((product) => (
            <div className="card" key={product.id} data-product-id={product.id}>
              <h3>
                <Link href={`/product/${product.id}`}>{product.name}</Link>
              </h3>
              <div className="price" data-testid={`price-${product.id}`}>
                ${product.price.toFixed(2)}
              </div>
              <p className="muted">{product.description}</p>
              <AddToCartButton
                productId={product.id}
                name={product.name}
                price={product.price}
              />
            </div>
          ))}
        </div>
      )}

      <div className="pager">
        {page > 1 ? (
          <Link href={buildHref({ q, sort, page: page - 1 })}>Previous</Link>
        ) : (
          <span className="muted">Previous</span>
        )}
        <span className="badge">
          Page {page} of {result.pages}
        </span>
        {page < result.pages ? (
          <Link href={buildHref({ q, sort, page: page + 1 })} data-testid="next-page">
            Next
          </Link>
        ) : (
          <span className="muted">Next</span>
        )}
      </div>
    </>
  );
}
