// types/index.ts
// TypeScript interfaces that mirror the API response shapes from Week 4.
// Having these here means if the API changes, TypeScript tells you
// exactly which components break — instead of finding out at runtime.

export interface TokenContribution {
  token: string;
  shap_value: number;
}

export interface Explanation {
  top_positive: TokenContribution[];
  top_negative: TokenContribution[];
  base_value: number;
  explanation_text: string;
}

export interface Ticket {
  ticket_id: string;
  subject: string;
  predicted_category: string;
  predicted_urgency: string;
  confidence: number;
  auto_routed: boolean;
  routed_to: string;
  source: string;
  created_at: string;
}

export interface TicketDetail extends Ticket {
  body: string;
  explanation: Explanation;
}

export interface TicketPredictionResponse {
  ticket_id: string;
  predicted_category: string;
  predicted_urgency: string;
  confidence: number;
  auto_routed: boolean;
  routed_to: string;
  explanation: Explanation;
  processing_time_ms: number;
  model_version: string;
  created_at: string;
}

export interface Metrics {
  total_tickets: number;
  auto_routed_count: number;
  auto_route_rate: number;
  avg_confidence: number;
  override_count: number;
  override_rate: number;
  tickets_by_category: Record<string, number>;
  tickets_by_urgency: Record<string, number>;
  model_version: string;
}

export interface SubmitTicketPayload {
  subject: string;
  body: string;
  source: string;
}

export interface OverridePayload {
  corrected_category: string;
  corrected_urgency: string;
  agent_id: string;
  correction_note?: string;
}

// The 6 categories and their display names
export const CATEGORIES = {
  billing: "Billing",
  bug_report: "Bug Report",
  feature_request: "Feature Request",
  account_access: "Account Access",
  performance: "Performance",
  general: "General",
} as const;

// Category → team name (mirrors the API routing rules)
export const CATEGORY_TEAMS: Record<string, string> = {
  billing: "Billing Team",
  bug_report: "Engineering Team",
  feature_request: "Product Team",
  account_access: "Account Support",
  performance: "Engineering Team",
  general: "General Queue",
};

// Urgency → display color
export const URGENCY_COLORS: Record<string, string> = {
  critical: "#dc2626",
  high: "#ea580c",
  normal: "#16a34a",
};

// Category → background color for badges
export const CATEGORY_COLORS: Record<string, string> = {
  billing: "#dbeafe",
  bug_report: "#fee2e2",
  feature_request: "#ede9fe",
  account_access: "#fef3c7",
  performance: "#d1fae5",
  general: "#f3f4f6",
};

export const CATEGORY_TEXT_COLORS: Record<string, string> = {
  billing: "#1d4ed8",
  bug_report: "#dc2626",
  feature_request: "#7c3aed",
  account_access: "#92400e",
  performance: "#065f46",
  general: "#374151",
};