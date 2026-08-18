
import React from "react";
import {
  CATEGORY_COLORS,
  CATEGORY_TEXT_COLORS,
  CATEGORIES,
  URGENCY_COLORS,
  Ticket,
} from "../types";

interface TicketCardProps {
  ticket: Ticket;
  isSelected: boolean;
  onClick: () => void;
}

const TicketCard: React.FC<TicketCardProps> = ({
  ticket,
  isSelected,
  onClick,
}) => {
  const confidencePct = Math.round(ticket.confidence * 100);

  // Confidence bar color — green above threshold, amber below
  const barColor =
    ticket.confidence >= 0.75 ? "#16a34a" : ticket.confidence >= 0.5 ? "#ea580c" : "#dc2626";

  const categoryLabel =
    CATEGORIES[ticket.predicted_category as keyof typeof CATEGORIES] ||
    ticket.predicted_category;

  return (
    <div
      onClick={onClick}
      style={{
        background: isSelected ? "#eff6ff" : "white",
        border: `1px solid ${isSelected ? "#3b82f6" : "#e5e7eb"}`,
        borderRadius: "8px",
        padding: "10px 12px",
        marginBottom: "8px",
        cursor: "pointer",
        transition: "border-color 0.15s, background 0.15s",
      }}
    >
      {/* Top row: ticket ID + category badge */}
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          marginBottom: "4px",
        }}
      >
        <span style={{ fontSize: "11px", color: "#9ca3af" }}>
          {ticket.ticket_id}
        </span>
        <span
          style={{
            fontSize: "10px",
            fontWeight: 600,
            padding: "2px 7px",
            borderRadius: "99px",
            background:
              CATEGORY_COLORS[ticket.predicted_category] || "#f3f4f6",
            color:
              CATEGORY_TEXT_COLORS[ticket.predicted_category] || "#374151",
          }}
        >
          {categoryLabel}
        </span>
      </div>

      {/* Subject line */}
      <div
        style={{
          fontSize: "13px",
          fontWeight: 500,
          color: "#111827",
          marginBottom: "4px",
          whiteSpace: "nowrap",
          overflow: "hidden",
          textOverflow: "ellipsis",
        }}
      >
        {ticket.subject || "(no subject)"}
      </div>

      {/* Routing destination */}
      <div
        style={{ fontSize: "11px", color: "#6b7280", marginBottom: "6px" }}
      >
        → {ticket.routed_to}
      </div>

      {/* Confidence bar + urgency dot */}
      <div
        style={{ display: "flex", alignItems: "center", gap: "6px" }}
      >
        {/* Urgency dot */}
        <span
          style={{
            width: "7px",
            height: "7px",
            borderRadius: "50%",
            background:
              URGENCY_COLORS[ticket.predicted_urgency] || "#16a34a",
            flexShrink: 0,
          }}
        />
        {/* Confidence bar */}
        <div
          style={{
            flex: 1,
            height: "4px",
            background: "#e5e7eb",
            borderRadius: "2px",
          }}
        >
          <div
            style={{
              width: `${confidencePct}%`,
              height: "4px",
              background: barColor,
              borderRadius: "2px",
              transition: "width 0.3s",
            }}
          />
        </div>
        <span style={{ fontSize: "10px", color: "#9ca3af", minWidth: "28px" }}>
          {confidencePct}%
        </span>
      </div>
    </div>
  );
};

export default TicketCard;