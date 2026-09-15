import Link from "next/link";
import { notFound } from "next/navigation";

import { AddToCartButton } from "@/app/_components/AddToCartButton";
import { findProduct } from "@/lib/products";

export default async function ProductPage({
  params,
}: {
  // Next 15 delivers route params as a promise.
  params: Promise<{ id: string }>;
}) {
  const { id } = await params;
  const product = findProduct(id);

  if (!product) {
    notFound();
  }

  return (
    <>
      <p>
        <Link href="/catalog">&larr; Back to catalog</Link>
      </p>

      <h1 data-testid="product-name">{product.name}</h1>
      <div className="price" data-testid="product-price">
        ${product.price.toFixed(2)}
      </div>
      <p className="muted">{product.description}</p>

      <table style={{ maxWidth: 420 }}>
        <tbody>
          <tr>
            <th>Category</th>
            <td>{product.category}</td>
          </tr>
          <tr>
            <th>SKU</th>
            <td>{product.id}</td>
          </tr>
          <tr>
            <th>In stock</th>
            <td>{product.stock}</td>
          </tr>
        </tbody>
      </table>

      <p style={{ marginTop: 18 }}>
        <AddToCartButton
          productId={product.id}
          name={product.name}
          price={product.price}
          quantity={1}
        />
      </p>
    </>
  );
}
