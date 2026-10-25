import { useEffect, useRef, useState } from "react";
import { api, Transaction } from "../api";

function decisionColor(decision: string) {
  if (decision === "block") return "text-signal border-signal/40 bg-signal/10";
  if (decision === "review") return "text-amber border-amber/40 bg-amber/10";
  return "text-safe border-safe/40 bg-safe/10";
}

export default function TransactionFeed() {
  const [txs, setTxs] = useState<Transaction[]>([]);
  const [running, setRunning] = useState(false);
  const pollRef = useRef<number | null>(null);

  const poll = async () => {
    try {
      const data = await api.recentTransactions(40);
      setTxs(data);
    } catch {
      /* API not reachable yet — feed stays empty until it is */
    }
  };

  useEffect(() => {
    poll();
    pollRef.current = window.setInterval(poll, 1500);
    return () => {
      if (pollRef.current) window.clearInterval(pollRef.current);
    };
  }, []);

  const toggleSimulation = async () => {
    if (running) {
      await api.stopSimulation();
      setRunning(false);
    } else {
      await api.startSimulation(800, 3000);
      setRunning(true);
    }
  };

  return (
    <div className="bg-panel border border-line rounded-lg flex flex-col h-full">
      <div className="flex items-center justify-between px-4 py-3 border-b border-line">
        <div className="flex items-center gap-2">
          <span className="text-xs uppercase tracking-widest text-muted">Live Feed</span>
          <span className={`h-1.5 w-1.5 rounded-full ${running ? "bg-safe animate-pulse" : "bg-muted"}`} />
        </div>
        <button
          onClick={toggleSimulation}
          className="text-xs font-mono px-3 py-1 rounded border border-line hover:border-amber hover:text-amber transition-colors"
        >
          {running ? "Stop replay" : "Start replay"}
        </button>
      </div>
      <div className="flex-1 overflow-y-auto font-mono text-xs">
        {txs.length === 0 && (
          <div className="p-6 text-muted text-center">
            No transactions yet. Press "Start replay" to stream held-out test transactions
            through the live model.
          </div>
        )}
        {txs.map((t) => (
          <div
            key={t.id}
            className="flex items-center justify-between px-4 py-2 border-b border-line/50 hover:bg-panel2"
          >
            <div className="flex items-center gap-3">
              <span className="text-muted w-16">#{t.id}</span>
              <span className="text-white w-20">{t.amount.toFixed(2)}</span>
              <span className="text-muted">{(t.fraud_probability * 100).toFixed(1)}%</span>
            </div>
            <div className="flex items-center gap-2">
              {t.rule_override_applied && (
                <span className="text-[10px] text-amber border border-amber/40 rounded px-1">
                  RULE
                </span>
              )}
              {t.actual_class === 1 && (
                <span className="text-[10px] text-signal">FRAUD</span>
              )}
              <span className={`px-2 py-0.5 rounded border text-[10px] uppercase ${decisionColor(t.decision)}`}>
                {t.decision}
              </span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
