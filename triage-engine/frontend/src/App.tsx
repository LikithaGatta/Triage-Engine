// App.tsx
// Root component. Handles navigation between Queue and Analytics views.
// Also shows the API connection status in the top bar.

import React, { useEffect, useState } from "react";
import { checkHealth } from "./api/client";
import Analytics from "./components/Analytics";
import QueueBoard from "./components/QueueBoard";

type View = "queue" | "analytics";

const App: React.FC = () => {
  const [view, setView]           = useState<View>("queue");
  const [apiStatus, setApiStatus] = useState<"checking" | "online" | "offline">("checking");

  // Check API health on mount
  useEffect(() => {
    checkHealth()
      .then(() => setApiStatus("online"))
      .catch(() => setApiStatus("offline"));
  }, []);

  const navTab = (id: View, label: string) => (
    <button
      onClick={() => setView(id)}
      style={{
        fontSize: "13px",
        padding: "5px 14px",
        borderRadius: "99px",
        border: "none",
        cursor: "pointer",
        background: view === id ? "#111827" : "transparent",
        color: view === id ? "white" : "#6b7280",
        fontWeight: view === id ? 500 : 400,
        transition: "background 0.15s",
      }}
    >
      {label}
    </button>
  );

  return (
    <div style={{ fontFamily: "'Inter', system-ui, sans-serif", background: "#f9fafb", minHeight: "100vh" }}>

      {/* Top bar */}
      <div style={{
        display: "flex", alignItems: "center", justifyContent: "space-between",
        padding: "10px 20px", background: "white",
        borderBottom: "1px solid #e5e7eb", position: "sticky", top: 0, zIndex: 10,
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: "16px" }}>
          {/* Logo */}
          <span style={{ fontSize: "15px", fontWeight: 600, color: "#111827" }}>
            TriageAI
          </span>

          {/* Nav tabs */}
          <div style={{ display: "flex", gap: "2px" }}>
            {navTab("queue", "Queue")}
            {navTab("analytics", "Analytics")}
          </div>
        </div>

        {/* API status indicator */}
        <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
          <span style={{
            width: "7px", height: "7px", borderRadius: "50%",
            background: apiStatus === "online" ? "#16a34a" : apiStatus === "offline" ? "#dc2626" : "#f59e0b",
            display: "inline-block",
          }} />
          <span style={{ fontSize: "12px", color: "#6b7280" }}>
            {apiStatus === "online" ? "API connected" : apiStatus === "offline" ? "API offline — start the server" : "Connecting..."}
          </span>
        </div>
      </div>

      {/* API offline warning */}
      {apiStatus === "offline" && (
        <div style={{
          background: "#fef2f2", border: "1px solid #fecaca",
          padding: "10px 20px", fontSize: "13px", color: "#dc2626",
          textAlign: "center",
        }}>
          API is offline. Run <code style={{ background: "#fee2e2", padding: "1px 6px", borderRadius: "4px" }}>python scripts/run_api.py</code> in your terminal to start it.
        </div>
      )}

      {/* Main content */}
      {view === "queue"     && <QueueBoard />}
      {view === "analytics" && <Analytics />}
    </div>
  );
};

export default App;