"use client";

import { useState } from "react";

interface Props {
  productId: string;
  name: string;
  price: number;
  quantity?: number;
}

/**
 * Adds a product to the cart through the public API.
 *
 * Goes through HTTP rather than calling the store directly so that the browser
 * and API test lanes exercise the same code path — otherwise a defect in this
 * route would be invisible to a browser-only agent.
 */
export function AddToCartButton({ productId, name, price, quantity = 1 }: Props) {
  const [status, setStatus] = useState<"idle" | "busy" | "done" | "error">("idle");
  const [message, setMessage] = useState("");

  async function add() {
    setStatus("busy");
    try {
      const response = await fetch("/api/cart", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ productId, name, price, quantity }),
      });
      const payload = (await response.json()) as { ok?: boolean; error?: string };
      if (response.ok && payload.ok) {
        setStatus("done");
        setMessage("Added to cart");
      } else {
        setStatus("error");
        setMessage(payload.error ?? `Request failed (${response.status})`);
      }
    } catch (error) {
      setStatus("error");
      setMessage(error instanceof Error ? error.message : "Network error");
    }
  }

  return (
    <div>
      <button
        type="button"
        className="primary"
        onClick={add}
        disabled={status === "busy"}
        aria-label={`Add ${name} to cart`}
      >
        {status === "busy" ? "Adding..." : "Add to cart"}
      </button>
      {message ? (
        <p className="muted" role="status" style={{ margin: "6px 0 0", fontSize: "0.85rem" }}>
          {message}
        </p>
      ) : null}
    </div>
  );
}
