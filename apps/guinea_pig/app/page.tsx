import Link from "next/link";

import { PRODUCTS } from "@/lib/products";

export default function HomePage() {
  const featured = PRODUCTS.slice(0, 3);

  return (
    <>
      <h1>Nimbus Supply</h1>
      <p className="muted">
        A small storefront used as the target application for testing. Every defect in this
        app is deliberate and declared in its source.
      </p>

      <h2>Featured</h2>
      <div className="grid">
        {featured.map((product) => (
          <div className="card" key={product.id}>
            <h3>
              <Link href={`/product/${product.id}`}>{product.name}</Link>
            </h3>
            <div className="price">${product.price.toFixed(2)}</div>
            <p className="muted">{product.description}</p>
          </div>
        ))}
      </div>

      <p style={{ marginTop: 24 }}>
        <Link href="/catalog">Browse the full catalog &rarr;</Link>
      </p>
    </>
  );
}
