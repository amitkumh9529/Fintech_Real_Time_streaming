import { useEffect, useState } from "react";
import { api, MetricsSummary, ModelInfo } from "../api";

export default function ImpactCalculator() {
  const [summary, setSummary] = useState<MetricsSummary | null>(null);
  const [info, setInfo] = useState<ModelInfo | null>(null);

  useEffect(() => {
    const poll = () => api.metricsSummary(2000).then(setSummary).catch(() => {});
    api.modelInfo().then(setInfo).catch(() => {});
    poll();
    const id = setInterval(poll, 3000);
    return () => clearInterval(id);
  }, []);

  const caught = summary?.fraud_amount_caught ?? 0;
  const missed = summary?.fraud_amount_missed ?? 0;
  const total = caught + missed;
  const captureRate = total > 0 ? (caught / total) * 100 : 0;

  const prAuc = info?.metrics?.pr_auc;

  return (
    <div className="bg-panel border border-line rounded-lg p-4 flex flex-col gap-4">
      <span className="text-xs uppercase tracking-widest text-muted">Replay Amount Summary</span>

      <div className="flex flex-col gap-1">
        <span className="text-[10px] uppercase tracking-widest text-muted">
          Fraud amount caught (this session)
        </span>
        <span className="text-3xl font-mono text-safe">
          {caught.toLocaleString(undefined, { maximumFractionDigits: 0 })}
        </span>
      </div>

      <div className="grid grid-cols-2 gap-3 text-xs font-mono">
        <div className="border border-line/60 rounded px-2 py-2">
          <div className="text-muted text-[10px]">Missed fraud amount</div>
          <div className="text-signal">{missed.toLocaleString(undefined, { maximumFractionDigits: 0 })}</div>
        </div>
        <div className="border border-line/60 rounded px-2 py-2">
          <div className="text-muted text-[10px]">Capture rate</div>
          <div className="text-white">{captureRate.toFixed(1)}%</div>
        </div>
      </div>

      <p className="text-[10px] text-muted leading-relaxed">
        Calculated as the sum of dataset <code>Amount</code> values for transactions correctly
        flagged as fraud during this replay session; it does not imply a currency or prevented
        losses. Validation-set PR-AUC:{" "}
        <span className="text-amber">{prAuc ? prAuc.toFixed(4) : "—"}</span>. See{" "}
        <code>eval_report.md</code> for the bootstrap evaluation and test-set amount summary.
      </p>
    </div>
  );
}
