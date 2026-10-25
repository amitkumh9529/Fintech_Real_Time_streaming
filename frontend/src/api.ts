const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

export interface Transaction {
  id: number;
  created_at: string;
  transaction_id: number | null;
  transaction_datetime: string | null;
  amount: number;
  actual_class: number | null;
  fraud_probability: number;
  decision: string;
  confidence: number;
  rule_override_applied: boolean;
  latency_ms: number;
}

export interface MetricsSummary {
  count: number;
  precision?: number;
  recall?: number;
  false_positive_rate?: number;
  true_positives?: number;
  false_positives?: number;
  false_negatives?: number;
  true_negatives?: number;
  fraud_amount_caught?: number;
  fraud_amount_missed?: number;
  avg_latency_ms?: number;
}

export interface ModelInfo {
  model_type: string;
  run_id: string;
  metrics: Record<string, number>;
  serving_since: string;
}

export interface DriftFeature {
  feature: string;
  ks_statistic: number;
  ks_pvalue: number;
  psi: number;
  flagged: boolean;
}

export interface DriftReport {
  psi_threshold: number;
  ks_pvalue_threshold: number;
  features: DriftFeature[];
  any_drift_detected: boolean;
}

async function getJSON<T>(path: string): Promise<T> {
  const res = await fetch(`${API_URL}${path}`);
  if (!res.ok) throw new Error(`${path} failed: ${res.status}`);
  return res.json();
}

export const api = {
  recentTransactions: (limit = 50) =>
    getJSON<Transaction[]>(`/dashboard/recent-transactions?limit=${limit}`),
  metricsSummary: (window = 500) =>
    getJSON<MetricsSummary>(`/dashboard/metrics-summary?window=${window}`),
  modelInfo: () => getJSON<ModelInfo>(`/model/info`),
  drift: () => getJSON<DriftReport>(`/dashboard/drift`),
  rocCurve: () => getJSON<{ roc: { fpr: number; tpr: number }[]; pr: { recall: number; precision: number }[] }>(
    `/dashboard/roc-curve`
  ),
  startSimulation: (speed = 1000, maxRows = 3000) =>
    fetch(`${API_URL}/simulation/start?speed_multiplier=${speed}&max_rows=${maxRows}`, {
      method: "POST",
    }).then((r) => r.json()),
  stopSimulation: () =>
    fetch(`${API_URL}/simulation/stop`, { method: "POST" }).then((r) => r.json()),
  simulationStatus: () => getJSON<{ running: boolean; speed_multiplier: number }>(
    `/simulation/status`
  ),
};
