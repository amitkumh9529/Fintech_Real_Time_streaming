import TransactionFeed from "./components/TransactionFeed";
import ModelMetrics from "./components/ModelMetrics";
import ROCCurve from "./components/ROCCurve";
import ImpactCalculator from "./components/ImpactCalculator";
import DriftMonitor from "./components/DriftMonitor";

export default function App() {
  return (
    <div className="min-h-screen bg-ink text-white">
      <header className="border-b border-line px-6 py-4 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="h-7 w-7 rounded bg-amber/20 border border-amber/40 flex items-center justify-center text-amber font-mono text-sm">
            L
          </div>
          <div>
            <h1 className="font-mono text-sm tracking-widest uppercase">Ledger</h1>
            <p className="text-[10px] text-muted">Fraud Detection Control Room</p>
          </div>
        </div>
        <div className="text-[10px] text-muted font-mono">
          Transaction Simulator · 1.75M tx · 0.837% fraud · Apr–Sep 2018
        </div>
      </header>

      <main className="p-6 grid grid-cols-1 lg:grid-cols-3 gap-4">
        <div className="lg:col-span-1 lg:row-span-2 h-[640px]">
          <TransactionFeed />
        </div>
        <div className="lg:col-span-2 grid grid-cols-1 md:grid-cols-2 gap-4">
          <ModelMetrics />
          <ImpactCalculator />
          <ROCCurve />
          <DriftMonitor />
        </div>
      </main>

      <footer className="px-6 py-4 text-[10px] text-muted border-t border-line">
        All metrics computed from real model evaluation on the held-out test set or from the
        live replay session — none are fabricated. See <code>eval_report.md</code> for the full
        report.
      </footer>
    </div>
  );
}
