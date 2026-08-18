
import React, { useCallback, useEffect, useState } from "react";
import { listTickets, submitTicket } from "../api/client";
import { Ticket } from "../types";
import TicketCard from "./TicketCard";
import TicketDetail from "./TicketDetail";

// The teams we route tickets to — each gets its own column
const QUEUES = [
  { id: "Billing Team",       label: "Billing",         color: "#dbeafe" },
  { id: "Engineering Team",   label: "Engineering",     color: "#fee2e2" },
  { id: "Product Team",       label: "Product",         color: "#ede9fe" },
  { id: "Account Support Team", label: "Account",       color: "#fef3c7" },
  { id: "Human Review Queue", label: "Human Review",    color: "#fce7f3" },
  { id: "General Support Queue", label: "General",      color: "#f3f4f6" },
];

const QueueBoard: React.FC = () => {
  const [tickets, setTickets]         = useState<Ticket[]>([]);
  const [selectedId, setSelectedId]   = useState<string | null>(null);
  const [submitting, setSubmitting]   = useState(false);
  const [subject, setSubject]         = useState("");
  const [body, setBody]               = useState("");
  const [lastResult, setLastResult]   = useState<string | null>(null);
  const [loading, setLoading]         = useState(true);

  const fetchTickets = useCallback(async () => {
    try {
      const data = await listTickets(undefined, 200);
      setTickets(data);
    } catch (e) {
      console.error("Failed to fetch tickets", e);
    } finally {
      setLoading(false);
    }
  }, []);

  // Poll for new tickets every 5 seconds
  useEffect(() => {
    fetchTickets();
    const interval = setInterval(fetchTickets, 5000);
    return () => clearInterval(interval);
  }, [fetchTickets]);

  const handleSubmit = async () => {
    if (!body.trim() || body.length < 5) return;
    setSubmitting(true);
    setLastResult(null);
    try {
      const result = await submitTicket({ subject, body, source: "dashboard" });
      setLastResult(
        `Ticket ${result.ticket_id} → ${result.routed_to} (${Math.round(result.confidence * 100)}% confidence)`
      );
      setSubject("");
      setBody("");
      // Refresh immediately after submit
      await fetchTickets();
      // Auto-select the new ticket
      setSelectedId(result.ticket_id);
    } catch (e) {
      setLastResult("Error submitting ticket. Is the API running?");
    } finally {
      setSubmitting(false);
    }
  };

  // Group tickets by their routed_to team
  const getQueueTickets = (queueId: string) =>
    tickets.filter(t => t.routed_to === queueId);

  return (
    <div style={{ display: "grid", gridTemplateColumns: "1fr 320px", height: "calc(100vh - 120px)", gap: 0 }}>

      {/* LEFT: Submit form + queue board */}
      <div style={{ overflowY: "auto", borderRight: "1px solid #e5e7eb" }}>

        {/* Submit form */}
        <div style={{ padding: "14px 16px", borderBottom: "1px solid #e5e7eb", background: "#f9fafb" }}>
          <div style={{ fontSize: "12px", fontWeight: 600, color: "#374151", marginBottom: "8px" }}>
            Submit a ticket for triage
          </div>
          <input
            value={subject}
            onChange={e => setSubject(e.target.value)}
            placeholder="Subject (optional)"
            style={{
              width: "100%", fontSize: "13px", border: "1px solid #d1d5db",
              borderRadius: "6px", padding: "6px 10px", marginBottom: "6px",
              boxSizing: "border-box", background: "white",
            }}
          />
          <textarea
            value={body}
            onChange={e => setBody(e.target.value)}
            placeholder="Describe the issue... (min 5 characters)"
            rows={3}
            style={{
              width: "100%", fontSize: "13px", border: "1px solid #d1d5db",
              borderRadius: "6px", padding: "6px 10px", marginBottom: "8px",
              boxSizing: "border-box", resize: "vertical", background: "white",
            }}
          />
          <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
            <button
              onClick={handleSubmit}
              disabled={submitting || body.length < 5}
              style={{
                fontSize: "13px", fontWeight: 500, padding: "6px 16px",
                borderRadius: "6px", border: "none",
                background: body.length >= 5 ? "#1d4ed8" : "#e5e7eb",
                color: body.length >= 5 ? "white" : "#9ca3af",
                cursor: body.length >= 5 ? "pointer" : "not-allowed",
              }}
            >
              {submitting ? "Triaging..." : "Triage ticket"}
            </button>
            {lastResult && (
              <span style={{ fontSize: "12px", color: "#16a34a" }}>{lastResult}</span>
            )}
          </div>
        </div>

        {/* Queue columns */}
        {loading ? (
          <div style={{ padding: "24px", color: "#9ca3af", fontSize: "13px" }}>Loading queues...</div>
        ) : (
          <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 0 }}>
            {QUEUES.map(queue => {
              const qTickets = getQueueTickets(queue.id);
              return (
                <div
                  key={queue.id}
                  style={{ borderRight: "1px solid #e5e7eb", borderBottom: "1px solid #e5e7eb", padding: "10px" }}
                >
                  {/* Queue header */}
                  <div style={{
                    display: "flex", alignItems: "center", justifyContent: "space-between",
                    marginBottom: "8px",
                  }}>
                    <span style={{ fontSize: "11px", fontWeight: 600, color: "#6b7280", textTransform: "uppercase", letterSpacing: "0.06em" }}>
                      {queue.label}
                    </span>
                    <span style={{
                      fontSize: "11px", background: "#f3f4f6", color: "#6b7280",
                      padding: "1px 7px", borderRadius: "99px",
                    }}>
                      {qTickets.length}
                    </span>
                  </div>

                  {/* Ticket cards */}
                  {qTickets.length === 0 ? (
                    <div style={{ fontSize: "12px", color: "#d1d5db", textAlign: "center", paddingTop: "12px" }}>
                      Empty
                    </div>
                  ) : (
                    qTickets.map(ticket => (
                      <TicketCard
                        key={ticket.ticket_id}
                        ticket={ticket}
                        isSelected={ticket.ticket_id === selectedId}
                        onClick={() => setSelectedId(ticket.ticket_id)}
                      />
                    ))
                  )}
                </div>
              );
            })}
          </div>
        )}
      </div>

      {/* RIGHT: Ticket detail panel */}
      <div style={{ overflowY: "auto" }}>
        {selectedId ? (
          <TicketDetail
            ticketId={selectedId}
            onOverride={fetchTickets}
          />
        ) : (
          <div style={{
            display: "flex", alignItems: "center", justifyContent: "center",
            height: "100%", color: "#9ca3af", fontSize: "13px", textAlign: "center", padding: "24px",
          }}>
            Select a ticket to see the AI decision and SHAP explanation
          </div>
        )}
      </div>
    </div>
  );
};

export default QueueBoard;