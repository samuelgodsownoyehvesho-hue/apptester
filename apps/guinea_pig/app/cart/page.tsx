"use client";

import { useCallback, useEffect, useState } from "react";

interface CartLine {
  productId: string;
  name: string;
  unitPrice: number;
  quantity: number;
}

interface CartSnapshot {
  lines: CartLine[];
  appliedCodes: string[];
  discountRate: number;
  subtotal: number;
  discount: number;
  total: number;
}

export default function CartPage() {
  const [cart, setCart] = useState<CartSnapshot | null>(null);
  const [code, setCode] = useState("");
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");

  const load = useCallback(async () => {
    const response = await fetch("/api/cart", { cache: "no-store" });
    setCart((await response.json()) as CartSnapshot);
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  async function remove(productId: string) {
    await fetch(`/api/cart?productId=${encodeURIComponent(productId)}`, { method: "DELETE" });
    setMessage("Item removed");
    await load();
  }

  async function applyCode() {
    setError("");
    const response = await fetch("/api/cart/discount", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ code }),
    });
    const payload = (await response.json()) as { ok?: boolean; error?: string };
    if (!response.ok || !payload.ok) {
      setError(payload.error ?? "Could not apply that code.");
    } else {
      setMessage(`Applied ${code.toUpperCase()}`);
      setCode("");
    }
    await load();
  }

  if (cart === null) {
    return <p className="muted">Loading cart...</p>;
  }

  return (
    <>
      <h1>Your cart</h1>

      {cart.lines.length === 0 ? (
        <p className="muted" data-testid="cart-empty">
          Your cart is empty.
        </p>
      ) : (
        <table>
          <thead>
            <tr>
              <th>Item</th>
              <th>Unit price</th>
              <th>Qty</th>
              <th>Line total</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {cart.lines.map((line) => (
              <tr key={line.productId} data-cart-line={line.productId}>
                <td>{line.name}</td>
                <td data-testid={`unit-${line.productId}`}>${line.unitPrice.toFixed(2)}</td>
                <td data-testid={`qty-${line.productId}`}>{line.quantity}</td>
                <td data-testid={`line-${line.productId}`}>
                  ${(line.unitPrice * line.quantity).toFixed(2)}
                </td>
                <td>
                  <button type="button" onClick={() => void remove(line.productId)}>
                    Remove
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
          <tfoot>
            <tr>
              <td colSpan={3}>Subtotal</td>
              <td data-testid="subtotal">${cart.subtotal.toFixed(2)}</td>
              <td />
            </tr>
            <tr>
              <td colSpan={3}>
                Discount{cart.appliedCodes.length ? ` (${cart.appliedCodes.join(", ")})` : ""}
              </td>
              <td data-testid="discount">-${cart.discount.toFixed(2)}</td>
              <td />
            </tr>
            <tr>
              <td colSpan={3}>Total</td>
              <td data-testid="cart-total">${cart.total.toFixed(2)}</td>
              <td />
            </tr>
          </tfoot>
        </table>
      )}

      <div className="toolbar">
        <label htmlFor="discount-code">Discount code</label>
        <input
          id="discount-code"
          value={code}
          onChange={(event) => setCode(event.target.value)}
          placeholder="SAVE10"
        />
        <button type="button" onClick={() => void applyCode()} data-testid="apply-discount">
          Apply
        </button>
      </div>

      {message ? (
        <p className="muted" role="status">
          {message}
        </p>
      ) : null}
      {error ? (
        <p role="alert" style={{ color: "var(--danger)" }}>
          {error}
        </p>
      ) : null}

      <p style={{ marginTop: 20 }}>
        <a className="primary" href="/checkout">
          Proceed to checkout
        </a>
      </p>
    </>
  );
}
