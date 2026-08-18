// api/client.ts
// All API calls live here. No fetch() calls scattered across components.
// If the API URL changes, you change it in one place.
// If an endpoint changes, TypeScript tells you every component that breaks.

import axios from "axios";
import {
  Metrics,
  OverridePayload,
  SubmitTicketPayload,
  Ticket,
  TicketDetail,
  TicketPredictionResponse,
} from "../types";

// Base URL — points to your running FastAPI server
const BASE_URL = "http://localhost:8000/api/v1";

// Axios instance with default config
// All requests automatically include base URL and JSON content type
const api = axios.create({
  baseURL: BASE_URL,
  headers: { "Content-Type": "application/json" },
  timeout: 10000, // 10 second timeout — fail fast if API is down
});


export const submitTicket = async (
  payload: SubmitTicketPayload
): Promise<TicketPredictionResponse> => {
  const response = await api.post<TicketPredictionResponse>("/tickets", payload);
  return response.data;
};

export const listTickets = async (
  category?: string,
  limit = 100
): Promise<Ticket[]> => {
  const params: Record<string, string | number> = { limit };
  if (category) params.category = category;
  const response = await api.get<Ticket[]>("/tickets", { params });
  return response.data;
};

export const getTicket = async (ticketId: string): Promise<TicketDetail> => {
  const response = await api.get<TicketDetail>(`/tickets/${ticketId}`);
  return response.data;
};

export const overrideTicket = async (
  ticketId: string,
  payload: OverridePayload
): Promise<void> => {
  await api.post(`/tickets/${ticketId}/override`, payload);
};

export const getMetrics = async (): Promise<Metrics> => {
  const response = await api.get<Metrics>("/metrics");
  return response.data;
};

export const checkHealth = async (): Promise<{ status: string }> => {
  const response = await api.get<{ status: string }>("/health");
  return response.data;
};