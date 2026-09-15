import type { Metadata } from "next";
import Link from "next/link";

import "./globals.css";

export const metadata: Metadata = {
  title: "Nimbus Supply — demo store",
  description: "Test target for the Crucible benchmark.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <header className="site">
          <Link className="brand" href="/">
            Nimbus Supply
          </Link>
          <nav>
            <Link href="/catalog">Catalog</Link>
          </nav>
          <nav>
            <Link href="/cart">Cart</Link>
          </nav>
          <nav>
            <Link href="/orders">Orders</Link>
          </nav>
          <nav>
            <Link href="/checkout">Checkout</Link>
          </nav>
        </header>

        <main>{children}</main>

        <footer className="site">
          <Link href="/about">About</Link>
          <Link href="/shipping">Shipping</Link>
          <span className="badge">demo target</span>
        </footer>
      </body>
    </html>
  );
}
