"use client";

import React, { useEffect, useState, useCallback } from "react";

/* ── Types ── */
interface Trade {
  trade_id: number;
  timestamp: string;
  asset: string;
  action: string;
  execution_price: number;
  predicted_volatility: number;
  regime: number;
  status: string;
  realized_pnl_pct: number | null;
}

interface PortfolioStats {
  total_realized_pnl_pct: number;
  win_rate: number;
  total_closed_trades: number;
  current_unrealized_pnl_pct: number;
}

const API = "http://142.93.189.92:3000";

/* ── Helpers ── */
function formatTime(ts: string): string {
  const d = new Date(ts);
  return d.toLocaleTimeString("en-US", {
    hour: "numeric",
    minute: "2-digit",
    second: "2-digit",
    hour12: true,
  });
}

function formatPnl(val: number | null | undefined): string {
  if (val === null || val === undefined) return "—";
  return `${val >= 0 ? "+" : ""}${val.toFixed(2)}%`;
}

function pnlGradient(val: number | null | undefined): string {
  if (val === null || val === undefined) return "text-slate-600";
  if (val > 0)
    return "bg-clip-text text-transparent bg-gradient-to-r from-emerald-400 to-cyan-400";
  if (val < 0)
    return "bg-clip-text text-transparent bg-gradient-to-r from-rose-500 to-orange-500";
  return "text-slate-400";
}

function pnlColorPlain(val: number | null | undefined): string {
  if (val === null || val === undefined) return "text-slate-600";
  if (val > 0) return "text-emerald-400";
  if (val < 0) return "text-rose-400";
  return "text-slate-500";
}

function actionBadge(action: string): string {
  switch (action) {
    case "BUY":
      return "bg-emerald-500/15 text-emerald-400 border border-emerald-500/30 shadow-[0_0_8px_rgba(52,211,153,0.15)]";
    case "SELL":
      return "bg-rose-500/15 text-rose-400 border border-rose-500/30 shadow-[0_0_8px_rgba(251,113,133,0.15)]";
    case "CASH":
      return "bg-slate-500/15 text-slate-400 border border-slate-500/30";
    case "HOLDING":
      return "bg-sky-500/15 text-sky-400 border border-sky-500/30 shadow-[0_0_8px_rgba(56,189,248,0.12)]";
    case "FLAT":
      return "bg-amber-500/15 text-amber-400 border border-amber-500/30";
    default:
      return "bg-slate-500/10 text-slate-500 border border-slate-500/20";
  }
}

function statusLED(status: string): string {
  switch (status) {
    case "OPEN":
      return "bg-emerald-500/15 text-emerald-400 border border-emerald-500/30 shadow-[0_0_6px_rgba(52,211,153,0.2)]";
    case "CLOSED":
      return "bg-slate-500/10 text-slate-500 border border-slate-600/30";
    default:
      return "bg-slate-500/10 text-slate-600 border border-slate-700/30";
  }
}

/* ══════════════════════════════════════════════════════════
   MAIN DASHBOARD (LiveOps)
   ══════════════════════════════════════════════════════════ */
export default function LiveOps() {
  const [trades, setTrades] = useState<Trade[]>([]);
  const [stats, setStats] = useState<PortfolioStats | null>(null);
  const [connected, setConnected] = useState(false);
  const [lastFetch, setLastFetch] = useState<Date | null>(null);

  const poll = useCallback(async () => {
    try {
      const [tradesRes, statsRes] = await Promise.all([
        fetch(`${API}/latest-state`),
        fetch(`${API}/portfolio-stats`),
      ]);
      if (!tradesRes.ok || !statsRes.ok) throw new Error();
      const tradesJson: Trade[] = await tradesRes.json();
      const statsJson: PortfolioStats = await statsRes.json();
      tradesJson.sort(
        (a, b) =>
          new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime()
      );
      setTrades(tradesJson);
      setStats(statsJson);
      setConnected(true);
      setLastFetch(new Date());
    } catch {
      setConnected(false);
    }
  }, []);

  useEffect(() => {
    poll();
    const id = setInterval(poll, 5000);
    return () => clearInterval(id);
  }, [poll]);

  return (
    <div className="min-h-[calc(100vh-4rem)] bg-gradient-to-br from-slate-950 via-[#0a0f1a] to-black">
      {/* Grid overlay */}
      <div className="fixed inset-0 pointer-events-none z-0 opacity-[0.03]"
        style={{
          backgroundImage:
            "linear-gradient(rgba(148,163,184,0.4) 1px, transparent 1px), linear-gradient(90deg, rgba(148,163,184,0.4) 1px, transparent 1px)",
          backgroundSize: "60px 60px",
        }}
      />

      <div className="relative z-10 px-4 sm:px-6 lg:px-8 py-6 max-w-[1440px] mx-auto">
        {/* ─── Header ─── */}
        <header className="flex items-center justify-between mb-8">
          <div>
            <h1 className="text-[20px] font-semibold text-white tracking-tight leading-none">
              LiveOps Dashboard
            </h1>
            <p className="text-[11px] text-slate-600 mt-1 tracking-[0.15em] uppercase font-mono">
              Phase 3 Live Execution
            </p>
          </div>
          <div className="flex items-center gap-5">
            {lastFetch && (
              <span className="text-[11px] font-mono text-slate-600 hidden sm:inline tabular-nums">
                {lastFetch.toLocaleTimeString("en-US", { hour12: true })}
              </span>
            )}
            <div className="flex items-center gap-2.5 px-4 py-2 rounded-full bg-white/[0.03] backdrop-blur-sm border border-white/[0.06]">
              <div className="relative flex items-center justify-center">
                <div className={`w-2 h-2 rounded-full ${connected ? "bg-emerald-400" : "bg-red-500"}`} />
                {connected && (
                  <div className="absolute inset-0 w-2 h-2 rounded-full bg-emerald-400 animate-ping opacity-75" />
                )}
              </div>
              <span className={`text-[11px] font-semibold tracking-wide ${connected ? "text-emerald-400" : "text-red-400"}`}>
                {connected ? "LIVE" : "OFFLINE"}
              </span>
            </div>
          </div>
        </header>

        {/* ═══════════════════════════════════════════
            SECTION 1: THE HUD
           ═══════════════════════════════════════════ */}
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 mb-6">
          <GlassCard
            label="Realized PnL"
            value={stats ? formatPnl(stats.total_realized_pnl_pct) : "—"}
            valueClass={stats ? pnlGradient(stats.total_realized_pnl_pct) : "text-slate-600"}
            sub="Cumulative closed trades"
            icon=""
          />
          <GlassCard
            label="Unrealized PnL"
            value={stats ? formatPnl(stats.current_unrealized_pnl_pct) : "—"}
            valueClass={stats ? pnlGradient(stats.current_unrealized_pnl_pct) : "text-slate-600"}
            sub={stats?.current_unrealized_pnl_pct !== 0 ? "Open position" : "No open position"}
            icon=""
          />
          <GlassCard
            label="Win Rate"
            value={stats ? `${stats.win_rate.toFixed(1)}%` : "—"}
            valueClass="bg-clip-text text-transparent bg-gradient-to-r from-sky-400 to-blue-400"
            sub={stats ? `${stats.total_closed_trades} closed trades` : "No data"}
            icon=""
          />
          <GlassCard
            label="Closed Trades"
            value={stats ? `${stats.total_closed_trades}` : "—"}
            valueClass="text-white"
            sub="Total round trips"
            icon=""
          />
        </div>

        {/* ═══════════════════════════════════════════
            SECTION 2: LIVE EXECUTION FEED
           ═══════════════════════════════════════════ */}
        <div className="rounded-2xl bg-white/[0.02] backdrop-blur-md border border-white/[0.06] overflow-hidden shadow-xl shadow-black/20">
          <div className="px-6 py-4 border-b border-white/[0.06] flex items-center justify-between">
            <div>
              <h2 className="text-[14px] font-semibold text-white tracking-tight">
                Live Execution Feed
              </h2>
              <p className="text-[11px] text-slate-600 mt-0.5 font-mono">
                Last {trades.length} executions · 5s polling
              </p>
            </div>
            <div className="flex items-center gap-4 text-[11px]">
              <span className="flex items-center gap-1.5 text-slate-500">
                <span className="w-2 h-2 rounded-full bg-emerald-400 shadow-[0_0_6px_rgba(52,211,153,0.4)]" /> BUY
              </span>
              <span className="flex items-center gap-1.5 text-slate-500">
                <span className="w-2 h-2 rounded-full bg-slate-400" /> CASH
              </span>
              <span className="flex items-center gap-1.5 text-slate-500">
                <span className="w-2 h-2 rounded-full bg-sky-400 shadow-[0_0_6px_rgba(56,189,248,0.3)]" /> HOLDING
              </span>
            </div>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-left">
              <thead>
                <tr className="border-b border-white/[0.04]">
                  <th className="px-6 py-3.5 text-[10px] font-semibold text-slate-500 uppercase tracking-[0.12em]">Timestamp</th>
                  <th className="px-6 py-3.5 text-[10px] font-semibold text-slate-500 uppercase tracking-[0.12em]">Action</th>
                  <th className="px-6 py-3.5 text-[10px] font-semibold text-slate-500 uppercase tracking-[0.12em] text-right">Price</th>
                  <th className="px-6 py-3.5 text-[10px] font-semibold text-slate-500 uppercase tracking-[0.12em] text-right">AI Volatility</th>
                  <th className="px-6 py-3.5 text-[10px] font-semibold text-slate-500 uppercase tracking-[0.12em] text-right">Regime</th>
                  <th className="px-6 py-3.5 text-[10px] font-semibold text-slate-500 uppercase tracking-[0.12em] text-right">Status</th>
                  <th className="px-6 py-3.5 text-[10px] font-semibold text-slate-500 uppercase tracking-[0.12em] text-right">PnL</th>
                </tr>
              </thead>
              <tbody>
                {trades.map((row, i) => (
                  <tr
                    key={row.trade_id}
                    className={`border-b border-white/[0.03] hover:bg-white/[0.03] transition-all duration-200 text-[13px] ${i % 2 === 1 ? "bg-white/[0.015]" : ""
                      }`}
                  >
                    <td className="px-6 py-3.5 font-mono text-[11px] text-slate-500 tabular-nums">
                      {formatTime(row.timestamp)}
                    </td>
                    <td className="px-6 py-3.5">
                      <span className={`inline-flex items-center px-2.5 py-1 rounded-md text-[10px] font-bold tracking-wider ${actionBadge(row.action)}`}>
                        {row.action}
                      </span>
                    </td>
                    <td className="px-6 py-3.5 text-right text-white font-medium font-mono text-[12px] tabular-nums">
                      ${row.execution_price.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                    </td>
                    <td className="px-6 py-3.5 text-right font-mono text-[11px] text-sky-400/60 tabular-nums">
                      {row.predicted_volatility.toFixed(5)}
                    </td>
                    <td className="px-6 py-3.5 text-right">
                      <span className={`inline-flex items-center gap-1.5 text-[10px] font-semibold tracking-wide ${row.regime === 0 ? "text-emerald-400/80" : "text-rose-400/80"
                        }`}>
                        <span className={`w-1.5 h-1.5 rounded-full ${row.regime === 0
                          ? "bg-emerald-400 shadow-[0_0_4px_rgba(52,211,153,0.5)]"
                          : "bg-rose-400 shadow-[0_0_4px_rgba(251,113,133,0.5)]"
                          }`} />
                        {row.regime === 0 ? "SAFE" : "DANGER"}
                      </span>
                    </td>
                    <td className="px-6 py-3.5 text-right">
                      <span className={`inline-flex items-center px-2 py-0.5 rounded text-[10px] font-bold tracking-wider ${statusLED(row.status)}`}>
                        {row.status}
                      </span>
                    </td>
                    <td className={`px-6 py-3.5 text-right font-mono text-[12px] font-semibold tabular-nums ${pnlColorPlain(row.realized_pnl_pct)}`}>
                      {formatPnl(row.realized_pnl_pct)}
                    </td>
                  </tr>
                ))}
                {trades.length === 0 && (
                  <tr>
                    <td colSpan={7} className="px-6 py-20 text-center text-[13px] text-slate-700">
                      <div className="flex flex-col items-center gap-3">
                        <div className="w-10 h-10 rounded-full bg-white/[0.03] border border-white/[0.06] flex items-center justify-center">
                          <div className="w-2.5 h-2.5 rounded-full bg-slate-600 animate-pulse" />
                        </div>
                        <span className="font-mono text-slate-600">Awaiting execution data…</span>
                      </div>
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </div>

        {/* ─── Footer ─── */}
        <footer className="mt-10 flex items-center justify-between text-[10px] text-slate-700 pb-6 font-mono tracking-wider uppercase">
          <span>Chronos v2.0 · Phase 3 LiveOps</span>
          <span>Polling {connected ? "active" : "paused"} · 5s interval</span>
        </footer>
      </div>
    </div>
  );
}

/* ── Glass Metric Card ── */
function GlassCard({
  label,
  value,
  valueClass = "text-white",
  sub,
  icon,
}: {
  label: string;
  value: string;
  valueClass?: string;
  sub?: React.ReactNode;
  icon?: string;
}) {
  return (
    <div className="group rounded-2xl bg-white/[0.02] backdrop-blur-md border border-white/[0.06] px-5 py-5 flex flex-col justify-between min-h-[120px] hover:bg-white/[0.04] hover:border-white/[0.1] transition-all duration-300 shadow-lg shadow-black/10">
      <div className="flex items-center justify-between">
        <span className="text-[10px] font-semibold text-slate-500 uppercase tracking-[0.12em]">
          {label}
        </span>
        {icon && <span className="text-[14px] opacity-40 group-hover:opacity-70 transition-opacity">{icon}</span>}
      </div>
      <div className="mt-auto">
        <span className={`text-[28px] font-bold leading-none font-mono tracking-tight ${valueClass}`}>
          {value}
        </span>
        {sub && (
          <p className="text-[10px] text-slate-600 mt-2 font-mono tracking-wide">{sub}</p>
        )}
      </div>
    </div>
  );
}
