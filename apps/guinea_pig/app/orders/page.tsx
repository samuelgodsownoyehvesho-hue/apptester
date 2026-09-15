const ORDERS = [
  { id: "NS-1001", placed: "2026-09-02", items: 3, total: 289.47, status: "Delivered" },
  { id: "NS-1002", placed: "2026-09-07", items: 1, total: 59.25, status: "Shipped" },
  { id: "NS-1003", placed: "2026-09-13", items: 2, total: 128.49, status: "Processing" },
];

export default function OrdersPage() {
  return (
    <>
      <h1>Orders</h1>
      <p className="muted">Recent orders for this account.</p>

      <table>
        <thead>
          <tr>
            <th>Order</th>
            <th>Placed</th>
            <th>Items</th>
            <th>Total</th>
            <th>Status</th>
          </tr>
        </thead>
        <tbody>
          {ORDERS.map((order) => (
            <tr key={order.id} data-order-id={order.id}>
              <td>{order.id}</td>
              <td>{order.placed}</td>
              <td>{order.items}</td>
              <td>${order.total.toFixed(2)}</td>
              <td>{order.status}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </>
  );
}
