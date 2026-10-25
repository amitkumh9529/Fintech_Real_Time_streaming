import { useEffect, useState } from "react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { api } from "../api";

export default function ROCCurve() {
  const [roc, setRoc] = useState<{ fpr: number; tpr: number }[]>([]);
  const [pr, setPr] = useState<{ recall: number; precision: number }[]>([]);
  const [tab, setTab] = useState<"roc" | "pr">("pr");

  useEffect(() => {
    api
      .rocCurve()
      .then((d) => {
        setRoc(d.roc);
        setPr(d.pr);
      })
      .catch(() => {});
  }, []);

  const data = tab === "roc" ? roc : pr;
  const xKey = tab === "roc" ? "fpr" : "recall";
  const yKey = tab === "roc" ? "tpr" : "precision";

  return (
    <div className="bg-panel border border-line rounded-lg p-4 flex flex-col gap-3">
      <div className="flex items-center justify-between">
        <span className="text-xs uppercase tracking-widest text-muted">
          {tab === "roc" ? "ROC Curve" : "Precision–Recall Curve"}
        </span>
        <div className="flex gap-1 text-[10px] font-mono">
          <button
            onClick={() => setTab("pr")}
            className={`px-2 py-1 rounded border ${tab === "pr" ? "border-amber text-amber" : "border-line text-muted"}`}
          >
            PR
          </button>
          <button
            onClick={() => setTab("roc")}
            className={`px-2 py-1 rounded border ${tab === "roc" ? "border-amber text-amber" : "border-line text-muted"}`}
          >
            ROC
          </button>
        </div>
      </div>
      <div className="h-56">
        {data.length === 0 ? (
          <div className="h-full flex items-center justify-center text-muted text-xs">
            Loading curve from real held-out test predictions…
          </div>
        ) : (
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={data} margin={{ left: -16, right: 8, top: 8, bottom: 0 }}>
              <CartesianGrid stroke="#232938" strokeDasharray="3 3" />
              <XAxis
                dataKey={xKey}
                stroke="#7C8699"
                fontSize={10}
                tickFormatter={(v) => v.toFixed(1)}
                domain={[0, 1]}
              />
              <YAxis stroke="#7C8699" fontSize={10} domain={[0, 1]} tickFormatter={(v) => v.toFixed(1)} />
              <Tooltip
                contentStyle={{ background: "#161B26", border: "1px solid #232938", fontSize: 11 }}
                labelFormatter={(v) => `${xKey}: ${Number(v).toFixed(3)}`}
              />
              <Line type="monotone" dataKey={yKey} stroke="#E8A33D" dot={false} strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>
        )}
      </div>
      <p className="text-[10px] text-muted">
        Computed from champion predictions on the chronological held-out test set —
        not from replayed API traffic.
      </p>
    </div>
  );
}
