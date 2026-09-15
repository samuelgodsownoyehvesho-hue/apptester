/**
 * Product catalog and query logic.
 *
 * Prices are chosen so that a string sort differs from a numeric sort
 * (`149.99` sorts before `19.99` lexicographically), which is what makes
 * `LEXICOGRAPHIC_SORT` detectable by a metamorphic check rather than by luck.
 *
 * Names deliberately mix casing so `CASE_SENSITIVE_SEARCH` is observable.
 */

import { bugOn } from "./bugs";

export interface Product {
  id: string;
  name: string;
  price: number;
  category: string;
  description: string;
  stock: number;
}

export const PRODUCTS: readonly Product[] = [
  { id: "p-01", name: "Laptop Pro 14", price: 149.99, category: "laptops", description: "14-inch aluminium chassis, 16GB memory.", stock: 12 },
  { id: "p-02", name: "Laptop Air 13", price: 89.99, category: "laptops", description: "Fanless, 18-hour battery.", stock: 7 },
  { id: "p-03", name: "laptop stand", price: 19.99, category: "accessories", description: "Adjustable aluminium riser.", stock: 40 },
  { id: "p-04", name: "Mechanical Keyboard", price: 129.5, category: "accessories", description: "Hot-swappable switches.", stock: 22 },
  { id: "p-05", name: "Wireless Mouse", price: 9.99, category: "accessories", description: "Silent switches, USB-C.", stock: 65 },
  { id: "p-06", name: "27-inch Monitor", price: 319.0, category: "displays", description: "4K IPS, 99% sRGB.", stock: 5 },
  { id: "p-07", name: "Ultrawide Monitor", price: 549.95, category: "displays", description: "34-inch curved panel.", stock: 3 },
  { id: "p-08", name: "USB-C Dock", price: 79.0, category: "accessories", description: "11 ports, 100W passthrough.", stock: 18 },
  { id: "p-09", name: "Noise Cancelling Headphones", price: 199.99, category: "audio", description: "Hybrid ANC, 30-hour battery.", stock: 14 },
  { id: "p-10", name: "Desk Microphone", price: 59.25, category: "audio", description: "Cardioid condenser, USB.", stock: 9 },
  { id: "p-11", name: "LAPTOP Sleeve 15", price: 24.5, category: "accessories", description: "Water-resistant felt.", stock: 31 },
  { id: "p-12", name: "Webcam 1080p", price: 44.99, category: "video", description: "Autofocus, privacy shutter.", stock: 16 },
  { id: "p-13", name: "Ergonomic Chair", price: 429.0, category: "furniture", description: "Lumbar support, 4D armrests.", stock: 4 },
  { id: "p-14", name: "Standing Desk", price: 699.99, category: "furniture", description: "Electric lift, 120kg capacity.", stock: 2 },
];

export type SortKey = "name" | "price_asc" | "price_desc";

export interface CatalogQuery {
  q?: string;
  sort?: SortKey;
  page?: number;
  perPage?: number;
}

export interface CatalogPage {
  items: Product[];
  total: number;
  page: number;
  perPage: number;
  pages: number;
}

export const DEFAULT_PER_PAGE = 6;

function applySearch(items: readonly Product[], q: string): Product[] {
  if (bugOn("CASE_SENSITIVE_SEARCH")) {
    // Defect: no case folding, so "laptop" misses "Laptop Pro 14".
    return items.filter((product) => product.name.includes(q));
  }
  const needle = q.toLowerCase();
  return items.filter((product) => product.name.toLowerCase().includes(needle));
}

function applySort(items: readonly Product[], sort: SortKey): Product[] {
  const copy = [...items];
  switch (sort) {
    case "price_asc":
      if (bugOn("LEXICOGRAPHIC_SORT")) {
        // Defect: comparing the string form, so 149.99 precedes 19.99.
        return copy.sort((a, b) => String(a.price).localeCompare(String(b.price)));
      }
      return copy.sort((a, b) => a.price - b.price);
    case "price_desc":
      return copy.sort((a, b) => b.price - a.price);
    case "name":
    default:
      return copy.sort((a, b) => a.name.localeCompare(b.name));
  }
}

function paginate(
  items: readonly Product[],
  page: number,
  perPage: number,
): { items: Product[]; offset: number } {
  const safePage = Number.isFinite(page) && page >= 1 ? Math.floor(page) : 1;
  const offset = (safePage - 1) * perPage;

  if (bugOn("PAGINATION_OVERLAP")) {
    // Defect: the window is shifted back by one, so the final item of each
    // page reappears at the top of the next.
    const shifted = Math.max(0, offset - 1);
    return { items: items.slice(shifted, shifted + perPage), offset: shifted };
  }

  return { items: items.slice(offset, offset + perPage), offset };
}

export function queryCatalog(query: CatalogQuery): CatalogPage {
  const perPage =
    Number.isFinite(query.perPage) && (query.perPage ?? 0) > 0
      ? Math.floor(query.perPage as number)
      : DEFAULT_PER_PAGE;

  let items: readonly Product[] = PRODUCTS;

  if (query.q && query.q.trim() !== "") {
    items = applySearch(items, query.q);
  }

  items = applySort(items, query.sort ?? "name");

  const page = query.page ?? 1;
  const { items: pageItems, offset } = paginate(items, page, perPage);

  return {
    items: pageItems,
    total: items.length,
    page: Math.floor(offset / perPage) + 1,
    perPage,
    pages: Math.max(1, Math.ceil(items.length / perPage)),
  };
}

export function findProduct(id: string): Product | undefined {
  return PRODUCTS.find((product) => product.id === id);
}
