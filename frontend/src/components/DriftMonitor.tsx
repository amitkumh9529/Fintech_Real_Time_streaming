import { useEffect, useState } from "react";
import { api, DriftReport } from "../api";

export default function DriftMonitor() {
  const [report, setReport] = useState<DriftReport | null>(null);

  useEffect(() => {
    api.drift().then(setReport).catch(() => {});
  }, []);

  return (
    <div className="bg-panel border border-line rounded-lg p-4 flex flex-col gap-3">
      <div className="flex items-center justify-between">
        <span className="text-xs uppercase tracking-widest text-muted">Offline Drift Check</span>
        {report && (
          <span
            className={`text-[10px] uppercase px-2 py-0.5 rounded border ${
              report.any_drift_detected
                ? "border-signal text-signal"
                : "border-safe text-safe"
            }`}
          >
            {report.any_drift_detected ? "Drift detected" : "Stable"}
          </span>
        )}
      </div>

      {!report ? (
        <div className="text-muted text-xs">Running KS-test / PSI against training distribution…</div>
      ) : (
        <table className="w-full text-[11px] font-mono">
          <thead>
            <tr className="text-muted text-left border-b border-line">
              <th className="py-1">Feature</th>
              <th>KS p-value</th>
              <th>PSI</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            {report.features.map((f) => (
              <tr key={f.feature} className="border-b border-line/40">
                <td className="py-1 text-white">{f.feature}</td>
                <td className={f.ks_pvalue < report.ks_pvalue_threshold ? "text-signal" : "text-muted"}>
                  {f.ks_pvalue.toFixed(4)}
                </td>
                <td className={f.psi > report.psi_threshold ? "text-signal" : "text-muted"}>
                  {f.psi.toFixed(3)}
                </td>
                <td>{f.flagged ? <span className="text-signal">⚠</span> : <span className="text-safe">✓</span>}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      <p className="text-[10px] text-muted">
        Compares stratified training and test samples; this dataset has no documented event time,
        so this is not live or temporal production drift. PSI &gt; {report?.psi_threshold ?? 0.2} or
        KS p-value &lt; {report?.ks_pvalue_threshold ?? 0.05} flags meaningful shift.
      </p>
    </div>
  );
}
