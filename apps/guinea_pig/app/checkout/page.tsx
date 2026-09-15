"use client";

import { useState } from "react";

interface CartSnapshot {
  lines: unknown[];
  total: number;
}

export default function CheckoutPage() {
  const [email, setEmail] = useState("");
  const [card, setCard] = useState("");
  const [status, setStatus] = useState("");
  const [error, setError] = useState("");

  async function submit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError("");
    setStatus("Placing order...");

    const response = await fetch("/api/checkout", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, card }),
    });
    const payload = (await response.json()) as { ok?: boolean; error?: string; orderId?: string };

    if (!response.ok || !payload.ok) {
      setError(payload.error ?? "Checkout failed.");
      setStatus("");
      return;
    }
    setStatus(`Order ${payload.orderId} placed. Thank you.`);
  }

  return (
    <>
      <h1>Checkout</h1>

      <form onSubmit={submit} style={{ maxWidth: 420 }}>
        {/*
          Defect (UNLABELED_CHECKOUT_INPUT): this field has no associated
          <label>. A placeholder is not an accessible name, so screen readers
          announce only "edit text".
        */}
        <p style={{ marginBottom: 4 }}>Email address</p>
        <input
          type="email"
          name="email"
          value={email}
          onChange={(event) => setEmail(event.target.value)}
          placeholder="you@example.com"
          style={{ width: "100%" }}
        />

        <p style={{ margin: "16px 0 4px" }}>
          <label htmlFor="card">Card number</label>
        </p>
        <input
          id="card"
          name="card"
          value={card}
          onChange={(event) => setCard(event.target.value)}
          placeholder="4242 4242 4242 4242"
          style={{ width: "100%" }}
        />

        <p style={{ marginTop: 18 }}>
          <button type="submit" className="primary" data-testid="place-order">
            Place order
          </button>
        </p>
      </form>

      {status ? (
        <p role="status" style={{ color: "var(--ok)" }}>
          {status}
        </p>
      ) : null}
      {error ? (
        <p role="alert" style={{ color: "var(--danger)" }}>
          {error}
        </p>
      ) : null}
    </>
  );
}
