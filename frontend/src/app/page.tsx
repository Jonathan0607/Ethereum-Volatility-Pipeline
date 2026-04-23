"use client";

import React, { useEffect, useState, useCallback } from "react";
import {
  AreaChart,
  Area,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  ReferenceDot,
} from "recharts";

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
}

interface BacktestPoint {
  timestamp: string;
  close: number;
  cumulative_market: number;
  cumulative_strategy: number;
  signal: number;
}

interface BacktestData {
  metrics: {
    market_return: number;
    strategy_return: number;
    sharpe: number;
    max_drawdown: number;
  };
  series: BacktestPoint[];
}

const API = "http://127.0.0.1:8000";

/* ── Custom Tooltips ── */
function PriceTooltip({ active, payload }: any) {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload as Trade;
  return (
    <div className="bg-[#111827]/95 border border-[#1e293b] rounded-lg px-4 py-3 shadow-xl backdrop-blur-sm">
      <p className="text-[11px] text-[#64748b] font-mono">
        {new Date(d.timestamp).toLocaleString()}
      </p>
      <p className="text-[15px] font-semibold text-white mt-1">
        ${d.execution_price.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
      </p>
      <div className="flex items-center gap-2 mt-1.5">
        <span className={`text-[11px] font-bold uppercase tracking-wider ${
          d.action === "BUY" ? "text-emerald-400" : d.action === "SELL" ? "text-rose-400" : "text-[#64748b]"
        }`}>
          {d.action}
        </span>
        <span className="text-[10px] text-[#475569]">|</span>
        <span className="text-[11px] font-mono text-[#94a3b8]">σ {d.predicted_volatility.toFixed(4)}</span>
      </div>
    </div>
  );
}

function BacktestTooltip({ active, payload }: any) {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload as BacktestPoint;
  const stratPct = ((d.cumulative_strategy - 1) * 100).toFixed(1);
  const mktPct = ((d.cumulative_market - 1) * 100).toFixed(1);
  return (
    <div className="bg-[#111827]/95 border border-[#1e293b] rounded-lg px-4 py-3 shadow-xl backdrop-blur-sm">
      <p className="text-[11px] text-[#64748b] font-mono">
        {new Date(d.timestamp).toLocaleDateString(undefined, { month: "short", day: "numeric", year: "numeric" })}
      </p>
      <div className="mt-1.5 space-y-1">
        <p className="text-[12px]">
          <span className="text-[#94a3b8]">Strategy</span>{" "}
          <span className={`font-semibold ${parseFloat(stratPct) >= 0 ? "text-emerald-400" : "text-rose-400"}`}>{stratPct}%</span>
        </p>
        <p className="text-[12px]">
          <span className="text-[#94a3b8]">Market</span>{" "}
          <span className={`font-semibold ${parseFloat(mktPct) >= 0 ? "text-emerald-400" : "text-rose-400"}`}>{mktPct}%</span>
        </p>
      </div>
      {d.signal === 1 && (
        <p className="text-[10px] text-emerald-500/80 mt-1.5 font-medium">● LONG POSITION</p>
      )}
    </div>
  );
}

/* ── Main Dashboard ── */
export default function Dashboard() {
  const [trades, setTrades] = useState<Trade[]>([]);
  const [backtest, setBacktest] = useState<BacktestData | null>(null);
  const [connected, setConnected] = useState(false);
  const [lastFetch, setLastFetch] = useState<Date | null>(null);

  /* Poll live trades */
  const poll = useCallback(async () => {
    try {
      const res = await fetch(`${API}/latest-state`);
      if (!res.ok) throw new Error();
      const json: Trade[] = await res.json();
      json.sort((a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime());
      setTrades(json);
      setConnected(true);
      setLastFetch(new Date());
    } catch {
      setConnected(false);
    }
  }, []);

  /* Fetch backtest once */
  useEffect(() => {
    (async () => {
      try {
        const res = await fetch(`${API}/backtest-data`);
        if (!res.ok) throw new Error();
        const json = await res.json();
        if (json.series) setBacktest(json);
      } catch {
        console.error("Failed to load backtest data");
      }
    })();
  }, []);

  useEffect(() => {
    poll();
    const id = setInterval(poll, 5000);
    return () => clearInterval(id);
  }, [poll]);

  const latest = trades.length > 0 ? trades[trades.length - 1] : null;
  const regime = latest?.regime ?? 0;
  const isSafe = regime === 0;

  /* Chart domains */
  const prices = trades.map((t) => t.execution_price);
  const yMin = prices.length ? Math.floor(Math.min(...prices) * 0.999) : 0;
  const yMax = prices.length ? Math.ceil(Math.max(...prices) * 1.001) : 100;
  const buys = trades.filter((t) => t.action === "BUY");
  const sells = trades.filter((t) => t.action === "SELL");

  const m = backtest?.metrics;

  return (
    <div className="relative z-10 min-h-screen px-6 py-6 max-w-[1400px] mx-auto">

      {/* ─── Header ─── */}
      <header className="flex items-center justify-between mb-8">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-blue-500 to-violet-600 flex items-center justify-center text-white font-bold text-sm shadow-lg shadow-blue-500/20">
            C
          </div>
          <div>
            <h1 className="text-[18px] font-semibold text-white tracking-tight leading-none">Chronos</h1>
            <p className="text-[11px] text-[#475569] mt-0.5 tracking-wide">ETH EXECUTION ENGINE</p>
          </div>
        </div>
        <div className="flex items-center gap-4">
          {lastFetch && <span className="text-[11px] font-mono text-[#334155]">{lastFetch.toLocaleTimeString()}</span>}
          <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-[#0f172a] border border-[#1e293b]">
            <div className="relative">
              <div className={`w-2 h-2 rounded-full ${connected ? "bg-emerald-400 pulse-dot" : "bg-red-500"}`} />
            </div>
            <span className={`text-[11px] font-medium ${connected ? "text-emerald-400/80" : "text-red-400/80"}`}>
              {connected ? "LIVE" : "OFFLINE"}
            </span>
          </div>
        </div>
      </header>

      {/* ─── Metric Strip ─── */}
      <div className="grid grid-cols-4 gap-3 mb-6">
        <MetricCard label="POSITION" value={latest?.action ?? "—"}
          valueClass={latest?.action === "BUY" ? "text-emerald-400" : latest?.action === "SELL" ? "text-rose-400" : "text-[#64748b]"}
          sub={latest ? `Trade #${latest.trade_id}` : undefined} />
        <MetricCard label="EXEC PRICE"
          value={latest ? `$${latest.execution_price.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}` : "—"}
          valueClass="text-white" sub="ETH/USDT" />
        <MetricCard label="AI VOLATILITY"
          value={latest ? latest.predicted_volatility.toFixed(4) : "—"}
          valueClass="text-sky-400" sub="σ predicted" />
        <MetricCard label="REGIME" value={latest ? (isSafe ? "SAFE" : "DANGER") : "—"}
          valueClass={isSafe ? "text-emerald-400" : "text-rose-400"}
          sub={latest ? (<span className="flex items-center gap-1.5">
            <span className={`inline-block w-1.5 h-1.5 rounded-full ${isSafe ? "bg-emerald-400" : "bg-rose-400"}`} />
            GMM Regime {regime}
          </span>) : undefined} />
      </div>

      {/* ─── Live Execution Chart ─── */}
      <div className="rounded-xl bg-[#0c1220] border border-[#14192a] p-5 mb-6">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h2 className="text-[13px] font-medium text-[#94a3b8]">Live Execution Price</h2>
            <p className="text-[11px] text-[#334155] mt-0.5">Last {trades.length} cycles · Hourly</p>
          </div>
          <div className="flex items-center gap-4 text-[11px]">
            <span className="flex items-center gap-1.5"><span className="w-2 h-2 rounded-full bg-emerald-400" /> BUY</span>
            <span className="flex items-center gap-1.5"><span className="w-2 h-2 rounded-full bg-rose-400" /> SELL</span>
          </div>
        </div>
        <div className="h-[280px] w-full">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={trades} margin={{ top: 8, right: 12, left: 0, bottom: 0 }}>
              <defs>
                <linearGradient id="priceGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#3b82f6" stopOpacity={0.15} />
                  <stop offset="100%" stopColor="#3b82f6" stopOpacity={0} />
                </linearGradient>
              </defs>
              <XAxis dataKey="timestamp" stroke="transparent"
                tick={{ fill: "#334155", fontSize: 11, fontFamily: "var(--font-geist-mono)" }}
                tickFormatter={(v) => { const d = new Date(v); return `${d.getHours()}:${d.getMinutes().toString().padStart(2, "0")}`; }}
                tickLine={false} axisLine={false} minTickGap={40} />
              <YAxis domain={[yMin, yMax]} stroke="transparent"
                tick={{ fill: "#334155", fontSize: 11, fontFamily: "var(--font-geist-mono)" }}
                tickFormatter={(v) => `$${v}`} tickLine={false} axisLine={false} width={65} />
              <Tooltip content={<PriceTooltip />} cursor={{ stroke: "#1e293b", strokeWidth: 1 }} />
              <Area type="monotone" dataKey="execution_price" stroke="#3b82f6" strokeWidth={2}
                fill="url(#priceGrad)" dot={false}
                activeDot={{ r: 4, fill: "#3b82f6", stroke: "#0c1220", strokeWidth: 2 }} />
              {buys.map((b) => (
                <ReferenceDot key={`b-${b.trade_id}`} x={b.timestamp} y={b.execution_price}
                  r={5} fill="#34d399" stroke="#0c1220" strokeWidth={2} />
              ))}
              {sells.map((s) => (
                <ReferenceDot key={`s-${s.trade_id}`} x={s.timestamp} y={s.execution_price}
                  r={5} fill="#fb7185" stroke="#0c1220" strokeWidth={2} />
              ))}
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* ─── Backtest Performance ─── */}
      {backtest && (
        <>
          {/* Backtest KPIs */}
          <div className="grid grid-cols-4 gap-3 mb-3">
            <MetricCard label="STRATEGY RETURN" value={m ? `${m.strategy_return > 0 ? "+" : ""}${m.strategy_return}%` : "—"}
              valueClass={m && m.strategy_return >= 0 ? "text-emerald-400" : "text-rose-400"} sub="Out-of-sample" />
            <MetricCard label="BUY & HOLD" value={m ? `${m.market_return > 0 ? "+" : ""}${m.market_return}%` : "—"}
              valueClass={m && m.market_return >= 0 ? "text-emerald-400" : "text-rose-400"} sub="ETH benchmark" />
            <MetricCard label="SHARPE RATIO" value={m ? `${m.sharpe}` : "—"}
              valueClass="text-amber-400" sub="Risk-adjusted" />
            <MetricCard label="MAX DRAWDOWN" value={m ? `${m.max_drawdown}%` : "—"}
              valueClass="text-rose-400" sub="Peak to trough" />
          </div>

          {/* Backtest Chart */}
          <div className="rounded-xl bg-[#0c1220] border border-[#14192a] p-5 mb-6">
            <div className="flex items-center justify-between mb-4">
              <div>
                <h2 className="text-[13px] font-medium text-[#94a3b8]">Backtest · Strategy vs Market</h2>
                <p className="text-[11px] text-[#334155] mt-0.5">
                  {backtest.series.length} data points · Out-of-sample test period
                </p>
              </div>
              <div className="flex items-center gap-4 text-[11px]">
                <span className="flex items-center gap-1.5">
                  <span className="w-3 h-[2px] rounded-full bg-blue-400 inline-block" /> Strategy
                </span>
                <span className="flex items-center gap-1.5">
                  <span className="w-3 h-[2px] rounded-full bg-[#475569] inline-block" /> Buy & Hold
                </span>
              </div>
            </div>
            <div className="h-[320px] w-full">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={backtest.series} margin={{ top: 8, right: 12, left: 0, bottom: 0 }}>
                  <XAxis dataKey="timestamp" stroke="transparent"
                    tick={{ fill: "#334155", fontSize: 11, fontFamily: "var(--font-geist-mono)" }}
                    tickFormatter={(v) => {
                      const d = new Date(v);
                      return d.toLocaleDateString(undefined, { month: "short", day: "numeric" });
                    }}
                    tickLine={false} axisLine={false} minTickGap={60} />
                  <YAxis stroke="transparent"
                    tick={{ fill: "#334155", fontSize: 11, fontFamily: "var(--font-geist-mono)" }}
                    tickFormatter={(v) => `${((v - 1) * 100).toFixed(0)}%`}
                    tickLine={false} axisLine={false} width={50}
                    domain={["dataMin", "dataMax"]} />
                  <Tooltip content={<BacktestTooltip />} cursor={{ stroke: "#1e293b", strokeWidth: 1 }} />
                  <Line type="monotone" dataKey="cumulative_market" stroke="#475569" strokeWidth={1.5}
                    dot={false} strokeDasharray="4 2" />
                  <Line type="monotone" dataKey="cumulative_strategy" stroke="#60a5fa" strokeWidth={2}
                    dot={false} activeDot={{ r: 3, fill: "#60a5fa", stroke: "#0c1220", strokeWidth: 2 }} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        </>
      )}

      {/* ─── Order Feed ─── */}
      <div className="rounded-xl bg-[#0c1220] border border-[#14192a] overflow-hidden">
        <div className="px-5 py-4 border-b border-[#14192a]">
          <h2 className="text-[13px] font-medium text-[#94a3b8]">Order Feed</h2>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-left">
            <thead>
              <tr className="border-b border-[#14192a] text-[11px] font-medium text-[#475569] uppercase tracking-wider">
                <th className="px-5 py-3">Time</th>
                <th className="px-5 py-3">Pair</th>
                <th className="px-5 py-3">Side</th>
                <th className="px-5 py-3 text-right">Price</th>
                <th className="px-5 py-3 text-right">Vol (σ)</th>
                <th className="px-5 py-3 text-right">Regime</th>
              </tr>
            </thead>
            <tbody>
              {[...trades].reverse().map((row) => (
                <tr key={row.trade_id} className="border-b border-[#14192a]/60 hover:bg-[#111827]/50 transition-colors text-[13px]">
                  <td className="px-5 py-3 font-mono text-[12px] text-[#475569]">
                    {new Date(row.timestamp).toLocaleString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" })}
                  </td>
                  <td className="px-5 py-3 text-[#94a3b8] font-medium">{row.asset}</td>
                  <td className="px-5 py-3">
                    <span className={`inline-block px-2 py-0.5 rounded text-[11px] font-bold tracking-wide ${
                      row.action === "BUY" ? "bg-emerald-500/10 text-emerald-400"
                        : row.action === "SELL" ? "bg-rose-500/10 text-rose-400"
                        : "bg-[#1e293b]/50 text-[#64748b]"
                    }`}>{row.action}</span>
                  </td>
                  <td className="px-5 py-3 text-right text-white font-medium font-mono text-[12px]">
                    ${row.execution_price.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                  </td>
                  <td className="px-5 py-3 text-right font-mono text-[12px] text-sky-400/70">
                    {row.predicted_volatility.toFixed(4)}
                  </td>
                  <td className="px-5 py-3 text-right">
                    <span className={`text-[11px] font-medium ${row.regime === 0 ? "text-emerald-400/70" : "text-rose-400/70"}`}>
                      {row.regime === 0 ? "SAFE" : "DANGER"}
                    </span>
                  </td>
                </tr>
              ))}
              {trades.length === 0 && (
                <tr><td colSpan={6} className="px-5 py-12 text-center text-[13px] text-[#334155]">Waiting for execution data…</td></tr>
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* ─── Footer ─── */}
      <footer className="mt-8 flex items-center justify-between text-[11px] text-[#1e293b] pb-4">
        <span>Chronos v1.0 · Phase 2 Sprint 2</span>
        <span>Polling {connected ? "active" : "paused"} · 5s interval</span>
      </footer>
    </div>
  );
}

/* ── Metric Card ── */
function MetricCard({ label, value, valueClass = "text-white", sub }: {
  label: string; value: string; valueClass?: string; sub?: React.ReactNode;
}) {
  return (
    <div className="rounded-xl bg-[#0c1220] border border-[#14192a] px-5 py-4 flex flex-col justify-between min-h-[100px] hover:border-[#1e293b] transition-colors">
      <span className="text-[11px] font-medium text-[#475569] uppercase tracking-wider">{label}</span>
      <div className="mt-auto">
        <span className={`text-[22px] font-semibold leading-none ${valueClass}`}>{value}</span>
        {sub && <p className="text-[11px] text-[#334155] mt-1.5">{sub}</p>}
      </div>
    </div>
  );
}
