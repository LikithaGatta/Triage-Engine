// components/Analytics.tsx
// Manager view — charts and metrics showing system performance.
// Uses Recharts for the bar charts.

import React, { useEffect, useState } from "react";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from "recharts";
import { getMetrics } from "../api/client";
import { Metrics, CATEGORIES, CATEGORY_COLORS } from "../types";

const Analytics: React.FC = () => {
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [loading, setLoading]  = useState(true);

  useEffect(() => {
    getMetrics()
      .then(setMetrics)
      .finally(() => setLoading(false));

    // Refresh every 10 seconds
    const interval = setInterval(() => getMetrics().then(setMetrics), 10000);
    return () => clearInterval(interval);
  }, []);

  if (loading) return <div style={{ padding: "24px", color: "#9ca3af", fontSize: "13px" }}>Loading analytics...</div>;
  if (!metrics)  return <div style={{ padding: "24px", color: "#dc2626", fontSize: "13px" }}>Failed to load metrics.</div>;

  // Transform category counts into chart data
  const categoryData = Object.entries(metrics.tickets_by_category).map(([cat, count]) => ({
    name: CATEGORIES[cat as keyof typeof CATEGORIES] || cat,
    count,
    fill: Object.values(CATEGORY_COLORS)[Object.keys(CATEGORY_COLORS).indexOf(cat)] || "#94a3b8",
  }));

  const urgencyData = Object.entries(metrics.tickets_by_urgency).map(([urg, count]) => ({
    name: urg.charAt(0).toUpperCase() + urg.slice(1),
    count,
    fill: urg === "critical" ? "#dc2626" : urg === "high" ? "#ea580c" : "#16a34a",
  }));

  // Stat card component defined inline for simplicity
  const StatCard = ({ label, value, sub, color = "#111827" }: { label: string; value: string; sub?: string; color?: string }) => (
    <div style={{
      background: "white", border: "1px solid #e5e7eb", borderRadius: "10px",
      padding: "14px 16px",
    }}>
      <div style={{ fontSize: "12px", color: "#6b7280", marginBottom: "4px" }}>{label}</div>
      <div style={{ fontSize: "28px", fontWeight: 600, color, marginBottom: "2px" }}>{value}</div>
      {sub && <div style={{ fontSize: "11px", color: "#9ca3af" }}>{sub}</div>}
    </div>
  );

  return (
    <div style={{ padding: "20px", overflowY: "auto" }}>

      {/* KPI cards */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: "12px", marginBottom: "24px" }}>
        <StatCard
          label="Total tickets"
          value={metrics.total_tickets.toLocaleString()}
          sub="All time"
        />
        <StatCard
          label="Auto-route rate"
          value={`${(metrics.auto_route_rate * 100).toFixed(1)}%`}
          sub={`${metrics.auto_routed_count} auto-routed`}
          color="#16a34a"
        />
        <StatCard
          label="Avg confidence"
          value={`${(metrics.avg_confidence * 100).toFixed(1)}%`}
          sub="Across all tickets"
          color="#1d4ed8"
        />
        <StatCard
          label="Override rate"
          value={`${(metrics.override_rate * 100).toFixed(1)}%`}
          sub={`${metrics.override_count} corrections`}
          color={metrics.override_rate > 0.1 ? "#dc2626" : "#374151"}
        />
      </div>

      {/* Charts row */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "16px", marginBottom: "24px" }}>

        {/* Category distribution chart */}
        <div style={{ background: "white", border: "1px solid #e5e7eb", borderRadius: "10px", padding: "16px" }}>
          <div style={{ fontSize: "13px", fontWeight: 600, color: "#111827", marginBottom: "14px" }}>
            Tickets by category
          </div>
          {categoryData.length > 0 ? (
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={categoryData} margin={{ top: 0, right: 8, bottom: 0, left: 0 }}>
                <XAxis dataKey="name" tick={{ fontSize: 10 }} />
                <YAxis tick={{ fontSize: 10 }} width={30} />
                <Tooltip />
                <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                  {categoryData.map((entry, index) => (
                    <Cell key={index} fill={entry.fill} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <div style={{ height: "200px", display: "flex", alignItems: "center", justifyContent: "center", color: "#d1d5db", fontSize: "13px" }}>
              No tickets yet — submit some from the Queue tab
            </div>
          )}
        </div>

        {/* Urgency distribution chart */}
        <div style={{ background: "white", border: "1px solid #e5e7eb", borderRadius: "10px", padding: "16px" }}>
          <div style={{ fontSize: "13px", fontWeight: 600, color: "#111827", marginBottom: "14px" }}>
            Tickets by urgency
          </div>
          {urgencyData.length > 0 ? (
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={urgencyData} margin={{ top: 0, right: 8, bottom: 0, left: 0 }}>
                <XAxis dataKey="name" tick={{ fontSize: 11 }} />
                <YAxis tick={{ fontSize: 10 }} width={30} />
                <Tooltip />
                <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                  {urgencyData.map((entry, index) => (
                    <Cell key={index} fill={entry.fill} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <div style={{ height: "200px", display: "flex", alignItems: "center", justifyContent: "center", color: "#d1d5db", fontSize: "13px" }}>
              No tickets yet
            </div>
          )}
        </div>
      </div>

      {/* Model info */}
      <div style={{ background: "white", border: "1px solid #e5e7eb", borderRadius: "10px", padding: "16px" }}>
        <div style={{ fontSize: "13px", fontWeight: 600, color: "#111827", marginBottom: "10px" }}>
          System info
        </div>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: "12px" }}>
          {[
            { label: "Model version", value: metrics.model_version },
            { label: "Confidence threshold", value: "75%" },
            { label: "Architecture", value: "TF-IDF + LogReg + SHAP" },
          ].map(item => (
            <div key={item.label} style={{ background: "#f9fafb", borderRadius: "6px", padding: "10px 12px" }}>
              <div style={{ fontSize: "11px", color: "#9ca3af", marginBottom: "3px" }}>{item.label}</div>
              <div style={{ fontSize: "13px", fontWeight: 500, color: "#111827" }}>{item.value}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default Analytics;