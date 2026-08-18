
import React, { useEffect, useState } from "react";
import { getTicket, overrideTicket } from "../api/client";
import { CATEGORIES, CATEGORY_COLORS, CATEGORY_TEXT_COLORS, TicketDetail as TicketDetailType, URGENCY_COLORS } from "../types";

interface Props {
  ticketId: string;
  onOverride: () => void;
}

const TicketDetail: React.FC<Props> = ({ ticketId, onOverride }) => {
  const [ticket, setTicket] = useState<TicketDetailType | null>(null);
  const [loading, setLoading] = useState(true);
  const [overriding, setOverriding] = useState(false);
  const [overrideCategory, setOverrideCategory] = useState("");
  const [overrideSent, setOverrideSent] = useState(false);

  useEffect(() => {
    setLoading(true);
    setOverrideSent(false);
    getTicket(ticketId)
      .then(setTicket)
      .finally(() => setLoading(false));
  }, [ticketId]);

  const handleOverride = async () => {
    if (!overrideCategory || !ticket) return;
    await overrideTicket(ticketId, {
      corrected_category: overrideCategory,
      corrected_urgency: "normal",
      agent_id: "agent_demo",
      correction_note: "Manual override from dashboard",
    });
    setOverrideSent(true);
    setOverriding(false);
    onOverride();
  };

  if (loading) {
    return (
      <div style={{ padding: "24px", color: "#6b7280", fontSize: "13px" }}>
        Loading ticket details...
      </div>
    );
  }

  if (!ticket) {
    return (
      <div style={{ padding: "24px", color: "#6b7280", fontSize: "13px" }}>
        Ticket not found.
      </div>
    );
  }

  const confPct = Math.round(ticket.confidence * 100);
  const categoryLabel = CATEGORIES[ticket.predicted_category as keyof typeof CATEGORIES] || ticket.predicted_category;

  return (
    <div style={{ padding: "16px", overflowY: "auto", height: "100%" }}>

      {/* Header */}
      <div style={{ marginBottom: "14px" }}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "6px" }}>
          <span style={{ fontSize: "12px", color: "#9ca3af" }}>{ticket.ticket_id}</span>
          <span style={{
            fontSize: "11px", fontWeight: 600, padding: "2px 9px", borderRadius: "99px",
            background: CATEGORY_COLORS[ticket.predicted_category] || "#f3f4f6",
            color: CATEGORY_TEXT_COLORS[ticket.predicted_category] || "#374151",
          }}>
            {categoryLabel}
          </span>
        </div>
        <div style={{ fontSize: "14px", fontWeight: 600, color: "#111827", marginBottom: "4px" }}>
          {ticket.subject || "(no subject)"}
        </div>
        <div style={{ fontSize: "12px", color: "#6b7280" }}>
          Source: {ticket.source} · {new Date(ticket.created_at).toLocaleTimeString()}
        </div>
      </div>

      {/* Body */}
      <div style={{
        background: "#f9fafb", border: "1px solid #e5e7eb", borderRadius: "6px",
        padding: "10px 12px", marginBottom: "14px", fontSize: "13px",
        color: "#374151", lineHeight: 1.6, maxHeight: "100px", overflowY: "auto",
      }}>
        {ticket.body}
      </div>

      {/* AI Decision */}
      <div style={{ marginBottom: "14px" }}>
        <div style={{ fontSize: "11px", fontWeight: 600, color: "#6b7280", textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: "8px" }}>
          AI Decision
        </div>

        <div style={{ display: "flex", gap: "6px", marginBottom: "8px", flexWrap: "wrap" }}>
          <span style={{
            fontSize: "12px", fontWeight: 500, padding: "3px 10px", borderRadius: "99px",
            background: CATEGORY_COLORS[ticket.predicted_category] || "#f3f4f6",
            color: CATEGORY_TEXT_COLORS[ticket.predicted_category] || "#374151",
          }}>
            {categoryLabel}
          </span>
          <span style={{
            fontSize: "12px", fontWeight: 500, padding: "3px 10px", borderRadius: "99px",
            background: "#f3f4f6", color: "#374151",
          }}>
            <span style={{
              display: "inline-block", width: "7px", height: "7px", borderRadius: "50%",
              background: URGENCY_COLORS[ticket.predicted_urgency] || "#16a34a",
              marginRight: "5px", verticalAlign: "middle",
            }} />
            {ticket.predicted_urgency}
          </span>
        </div>

        {/* Confidence bar */}
        <div style={{ fontSize: "12px", color: "#6b7280", marginBottom: "4px" }}>
          Confidence: {confPct}%
          {ticket.auto_routed
            ? <span style={{ color: "#16a34a", marginLeft: "8px" }}>✓ Auto-routed</span>
            : <span style={{ color: "#ea580c", marginLeft: "8px" }}>⚠ Human review</span>}
        </div>
        <div style={{ height: "6px", background: "#e5e7eb", borderRadius: "3px", marginBottom: "8px" }}>
          <div style={{
            width: `${confPct}%`, height: "6px", borderRadius: "3px",
            background: ticket.confidence >= 0.75 ? "#16a34a" : "#ea580c",
            transition: "width 0.4s",
          }} />
        </div>
        <div style={{ fontSize: "12px", color: "#374151" }}>
          Routed to: <strong>{ticket.routed_to}</strong>
        </div>
      </div>

      {/* SHAP Explanation */}
      <div style={{ marginBottom: "14px" }}>
        <div style={{ fontSize: "11px", fontWeight: 600, color: "#6b7280", textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: "8px" }}>
          Why this category?
        </div>

        <div style={{ fontSize: "12px", color: "#374151", marginBottom: "8px", lineHeight: 1.5, background: "#f0f9ff", borderRadius: "6px", padding: "8px 10px", border: "1px solid #bae6fd" }}>
          {ticket.explanation.explanation_text}
        </div>

        {/* Positive tokens */}
        {ticket.explanation.top_positive.length > 0 && (
          <div style={{ marginBottom: "8px" }}>
            <div style={{ fontSize: "11px", color: "#166534", fontWeight: 500, marginBottom: "4px" }}>
              ↑ Signals toward {categoryLabel}
            </div>
            <div style={{ display: "flex", flexWrap: "wrap", gap: "4px" }}>
              {ticket.explanation.top_positive.map((t, i) => (
                <span key={i} style={{
                  fontSize: "11px", fontWeight: 500, padding: "2px 8px",
                  borderRadius: "99px", background: "#dcfce7", color: "#166534",
                }}>
                  "{t.token}" +{t.shap_value.toFixed(3)}
                </span>
              ))}
            </div>
          </div>
        )}

        {/* Negative tokens */}
        {ticket.explanation.top_negative.length > 0 && (
          <div>
            <div style={{ fontSize: "11px", color: "#991b1b", fontWeight: 500, marginBottom: "4px" }}>
              ↓ Competing signals
            </div>
            <div style={{ display: "flex", flexWrap: "wrap", gap: "4px" }}>
              {ticket.explanation.top_negative.map((t, i) => (
                <span key={i} style={{
                  fontSize: "11px", fontWeight: 500, padding: "2px 8px",
                  borderRadius: "99px", background: "#fee2e2", color: "#991b1b",
                }}>
                  "{t.token}" {t.shap_value.toFixed(3)}
                </span>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Override section */}
      <div style={{ borderTop: "1px solid #e5e7eb", paddingTop: "12px" }}>
        <div style={{ fontSize: "11px", fontWeight: 600, color: "#6b7280", textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: "8px" }}>
          Override Routing
        </div>

        {overrideSent ? (
          <div style={{ fontSize: "12px", color: "#16a34a", background: "#dcfce7", padding: "8px 10px", borderRadius: "6px" }}>
            ✓ Correction recorded. This feeds into model retraining.
          </div>
        ) : overriding ? (
          <div style={{ display: "flex", gap: "6px", alignItems: "center" }}>
            <select
              value={overrideCategory}
              onChange={e => setOverrideCategory(e.target.value)}
              style={{ fontSize: "12px", border: "1px solid #d1d5db", borderRadius: "6px", padding: "4px 8px", flex: 1, background: "white" }}
            >
              <option value="">Select category...</option>
              {Object.entries(CATEGORIES).map(([val, label]) => (
                <option key={val} value={val}>{label}</option>
              ))}
            </select>
            <button
              onClick={handleOverride}
              disabled={!overrideCategory}
              style={{
                fontSize: "12px", padding: "4px 10px", borderRadius: "6px",
                background: overrideCategory ? "#1d4ed8" : "#e5e7eb",
                color: overrideCategory ? "white" : "#9ca3af",
                border: "none", cursor: overrideCategory ? "pointer" : "not-allowed",
              }}
            >
              Save
            </button>
            <button
              onClick={() => setOverriding(false)}
              style={{ fontSize: "12px", padding: "4px 10px", borderRadius: "6px", background: "transparent", border: "1px solid #d1d5db", cursor: "pointer", color: "#374151" }}
            >
              Cancel
            </button>
          </div>
        ) : (
          <button
            onClick={() => setOverriding(true)}
            style={{
              fontSize: "12px", padding: "5px 12px", borderRadius: "6px",
              background: "transparent", border: "1px solid #d1d5db",
              cursor: "pointer", color: "#374151",
            }}
          >
            Correct this routing
          </button>
        )}
      </div>
    </div>
  );
};

export default TicketDetail;