export const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

async function request(path, options = {}) {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `Request failed (${res.status})`);
  }
  return res.json();
}

export const api = {
  edaSummary: () => request("/eda/summary"),
  edaInsights: () => request("/eda/insights"),
  predict: (sk_id_curr) =>
    request("/predict", { method: "POST", body: JSON.stringify({ sk_id_curr }) }),
  explain: (sk_id_curr) =>
    request("/explain", { method: "POST", body: JSON.stringify({ sk_id_curr }) }),
  rules: () => request("/rules"),
  chat: (question, history) =>
    request("/chat", { method: "POST", body: JSON.stringify({ question, history }) }),
};
