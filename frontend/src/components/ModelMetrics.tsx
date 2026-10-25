import { useEffect, useState } from "react";
import { api, MetricsSummary, ModelInfo } from "../api";

function Stat({ label, value, accent }: { label: string; value: string; accent?: string }) {
  return (
    <div className="flex flex-col gap-1">
      <span className="text-[10px] uppercase tracking-widest text-muted">{label}</span>
      <span className={`text-2xl font-mono ${accent || "text-white"}`}>{value}</span>
    </div>
  );
}

export default function ModelMetrics() {
  const [summary, setSummary] = useState<MetricsSummary | null>(null);
  const [info, setInfo] = useState<ModelInfo | null>(null);

  useEffect(() => {
    const poll = async () => {
      try {
        setSummary(await api.metricsSummary(500));
      } catch {}
    };
    api.modelInfo().then(setInfo).catch(() => {});
    poll();
    const id = setInterval(poll, 2000);
    return () => clearInterval(id);
  }, []);

  const pct = (v?: number) => (v !== undefined ? `${(v * 100).toFixed(1)}%` : "—");

  return (
    <div className="bg-panel border border-line rounded-lg p-4 flex flex-col gap-4">
      <div className="flex items-center justify-between">
        <span className="text-xs uppercase tracking-widest text-muted">Champion Model</span>
        <span className="text-xs font-mono text-amber">
          {info ? `${info.model_type} · ${info.run_id.slice(0, 8)}` : "loading…"}
        </span>
      </div>

      <div className="grid grid-cols-2 gap-4">
        <Stat label="Live precision" value={pct(summary?.precision)} accent="text-safe" />
        <Stat label="Live recall" value={pct(summary?.recall)} accent="text-safe" />
        <Stat label="False positive rate" value={pct(summary?.false_positive_rate)} accent="text-signal" />
        <Stat label="Avg latency" value={summary?.avg_latency_ms ? `${summary.avg_latency_ms.toFixed(1)}ms` : "—"} />
      </div>

      <div className="border-t border-line pt-3">
        <span className="text-[10px] uppercase tracking-widest text-muted">Validation metrics (eval set)</span>
        <div className="grid grid-cols-3 gap-2 mt-2 font-mono text-xs">
          {info && Object.entries(info.metrics).map(([k, v]) => (
            <div key={k} className="flex justify-between border border-line/60 rounded px-2 py-1">
              <span className="text-muted">{k}</span>
              <span className="text-white">{typeof v === "number" ? v.toFixed(4) : v}</span>
            </div>
          ))}
        </div>
      </div>

      <div className="text-[10px] text-muted font-mono">
        Sample window: last {summary?.count ?? 0} scored transactions
      </div>
    </div>
  );
}
