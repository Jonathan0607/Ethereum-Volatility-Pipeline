"use client";

import React, { useEffect, useState } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";

/* ── Types ── */
interface BacktestPoint {
  timestamp: string;
  close: number;
  cumulative_market: number;
  cumulative_strategy: number;
  signal: number;
  position_size: number;
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

const API = "http://142.93.189.92:8000";

/* ── Tooltip ── */
function BacktestTooltip({ active, payload }: any) {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload as BacktestPoint;
  const stratPct = ((d.cumulative_strategy - 1) * 100).toFixed(1);
  const mktPct = ((d.cumulative_market - 1) * 100).toFixed(1);
  return (
    <div className="bg-black/80 backdrop-blur-xl border border-white/10 rounded-xl px-5 py-3.5 shadow-2xl shadow-black/50">
      <p className="text-[10px] text-slate-500 font-mono uppercase tracking-wider">
        {new Date(d.timestamp).toLocaleDateString(undefined, {
          month: "short",
          day: "numeric",
          year: "numeric",
        })}
      </p>
      <div className="mt-2 space-y-1.5">
        <div className="flex items-center justify-between gap-6">
          <span className="text-[11px] text-slate-400">AI Strategy</span>
          <span
            className={`text-[13px] font-bold font-mono ${parseFloat(stratPct) >= 0 ? "text-emerald-400" : "text-rose-400"
              }`}
          >
            {parseFloat(stratPct) >= 0 ? "+" : ""}
            {stratPct}%
          </span>
        </div>
        <div className="flex items-center justify-between gap-6">
          <span className="text-[11px] text-slate-400">Buy & Hold</span>
          <span
            className={`text-[13px] font-bold font-mono ${parseFloat(mktPct) >= 0 ? "text-emerald-400" : "text-rose-400"
              }`}
          >
            {parseFloat(mktPct) >= 0 ? "+" : ""}
            {mktPct}%
          </span>
        </div>
      </div>
    </div>
  );
}

export default function BacktestPage() {
  const [backtest, setBacktest] = useState<BacktestData | null>(null);

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

  const m = backtest?.metrics;

  return (
    <div className="min-h-[calc(100vh-4rem)] bg-gradient-to-br from-slate-950 via-[#0a0f1a] to-black">
      {/* Grid overlay */}
      <div
        className="fixed inset-0 pointer-events-none z-0 opacity-[0.03]"
        style={{
          backgroundImage:
            "linear-gradient(rgba(148,163,184,0.4) 1px, transparent 1px), linear-gradient(90deg, rgba(148,163,184,0.4) 1px, transparent 1px)",
          backgroundSize: "60px 60px",
        }}
      />

      <div className="relative z-10 px-4 sm:px-6 lg:px-8 py-6 max-w-[1440px] mx-auto">
        <header className="mb-8">
          <h1 className="text-[20px] font-semibold text-white tracking-tight leading-none">
            Historical Backtest
          </h1>
          <p className="text-[11px] text-slate-600 mt-1 tracking-[0.15em] uppercase font-mono">
            Out-of-sample Performance vs Market
          </p>
        </header>

        {backtest ? (
          <>
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 mb-6">
              <GlassCard
                label="Strategy Return"
                value={m ? `${m.strategy_return > 0 ? "+" : ""}${m.strategy_return}%` : "—"}
                valueClass={
                  m && m.strategy_return >= 0
                    ? "bg-clip-text text-transparent bg-gradient-to-r from-emerald-400 to-cyan-400"
                    : "bg-clip-text text-transparent bg-gradient-to-r from-rose-500 to-orange-500"
                }
                sub="Out-of-sample"
              />
              <GlassCard
                label="Buy & Hold"
                value={m ? `${m.market_return > 0 ? "+" : ""}${m.market_return}%` : "—"}
                valueClass={
                  m && m.market_return >= 0
                    ? "bg-clip-text text-transparent bg-gradient-to-r from-emerald-400 to-cyan-400"
                    : "bg-clip-text text-transparent bg-gradient-to-r from-rose-500 to-orange-500"
                }
                sub="ETH benchmark"
              />
              <GlassCard
                label="Sharpe Ratio"
                value={m ? `${m.sharpe}` : "—"}
                valueClass="bg-clip-text text-transparent bg-gradient-to-r from-amber-400 to-yellow-300"
                sub="Risk-adjusted"
              />
              <GlassCard
                label="Max Drawdown"
                value={m ? `${m.max_drawdown}%` : "—"}
                valueClass="bg-clip-text text-transparent bg-gradient-to-r from-rose-500 to-orange-500"
                sub="Peak to trough"
              />
            </div>

            <div className="rounded-2xl bg-white/[0.02] backdrop-blur-md border border-white/[0.06] p-6 shadow-xl shadow-black/20 flex flex-col h-[650px]">
              <div className="flex items-center justify-between mb-5 shrink-0">
                <div>
                  <h2 className="text-[14px] font-semibold text-white tracking-tight">
                    Equity Curve
                  </h2>
                  <p className="text-[11px] text-slate-600 mt-0.5 font-mono">
                    {backtest.series.length} data points
                  </p>
                </div>
                <div className="flex items-center gap-5 text-[11px]">
                  <span className="flex items-center gap-2 text-slate-400">
                    <span className="w-4 h-[2px] rounded-full bg-gradient-to-r from-cyan-400 to-emerald-400 inline-block" />
                    AI Strategy
                  </span>
                  <span className="flex items-center gap-2 text-slate-500">
                    <span
                      className="w-4 h-[2px] rounded-full bg-slate-500 inline-block opacity-50"
                      style={{
                        backgroundImage:
                          "repeating-linear-gradient(90deg, #64748b 0, #64748b 4px, transparent 4px, transparent 8px)",
                      }}
                    />
                    Buy &amp; Hold
                  </span>
                </div>
              </div>

              <div className="w-full flex-1 min-h-0">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart
                    data={backtest.series}
                    margin={{ top: 10, right: 10, left: 0, bottom: 0 }}
                  >
                    <XAxis
                      dataKey="timestamp"
                      stroke="transparent"
                      tick={{
                        fill: "#475569",
                        fontSize: 10,
                        fontFamily: "var(--font-geist-mono)",
                      }}
                      tickFormatter={(v) =>
                        new Date(v).toLocaleDateString(undefined, {
                          month: "short",
                          day: "numeric",
                          year: "2-digit"
                        })
                      }
                      tickLine={false}
                      axisLine={false}
                      minTickGap={60}
                    />
                    <YAxis
                      stroke="transparent"
                      tick={{
                        fill: "#475569",
                        fontSize: 10,
                        fontFamily: "var(--font-geist-mono)",
                      }}
                      tickFormatter={(v) => `${((v - 1) * 100).toFixed(0)}%`}
                      tickLine={false}
                      axisLine={false}
                      width={50}
                      domain={["dataMin", "dataMax"]}
                    />
                    <Tooltip
                      content={<BacktestTooltip />}
                      cursor={{ stroke: "rgba(255,255,255,0.06)", strokeWidth: 1 }}
                    />
                    <Legend
                      verticalAlign="top"
                      align="right"
                      height={0}
                      wrapperStyle={{ display: "none" }}
                    />
                    <Line
                      type="monotone"
                      dataKey="cumulative_market"
                      name="Buy & Hold ETH"
                      stroke="#64748b"
                      strokeWidth={1.5}
                      dot={false}
                      strokeDasharray="5 5"
                      strokeOpacity={0.6}
                    />
                    <Line
                      type="monotone"
                      dataKey="cumulative_strategy"
                      name="AI Strategy"
                      stroke="#38bdf8"
                      strokeWidth={2.5}
                      dot={false}
                      activeDot={{
                        r: 4,
                        fill: "#38bdf8",
                        stroke: "rgba(56,189,248,0.3)",
                        strokeWidth: 6,
                      }}
                      filter="url(#glow)"
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>

              {/* SVG filter for glow effect */}
              <svg width="0" height="0" className="absolute">
                <defs>
                  <filter id="glow">
                    <feGaussianBlur stdDeviation="3" result="coloredBlur" />
                    <feMerge>
                      <feMergeNode in="coloredBlur" />
                      <feMergeNode in="SourceGraphic" />
                    </feMerge>
                  </filter>
                </defs>
              </svg>
            </div>
          </>
        ) : (
          <div className="flex items-center justify-center h-[50vh]">
            <div className="w-10 h-10 rounded-full bg-white/[0.03] border border-white/[0.06] flex items-center justify-center">
              <div className="w-2.5 h-2.5 rounded-full bg-slate-600 animate-pulse" />
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

function GlassCard({
  label,
  value,
  valueClass = "text-white",
  sub,
}: {
  label: string;
  value: string;
  valueClass?: string;
  sub?: string;
}) {
  return (
    <div className="group rounded-2xl bg-white/[0.02] backdrop-blur-md border border-white/[0.06] px-5 py-5 flex flex-col justify-between min-h-[120px] hover:bg-white/[0.04] hover:border-white/[0.1] transition-all duration-300 shadow-lg shadow-black/10">
      <span className="text-[10px] font-semibold text-slate-500 uppercase tracking-[0.12em]">
        {label}
      </span>
      <div className="mt-auto">
        <span
          className={`text-[28px] font-bold leading-none font-mono tracking-tight ${valueClass}`}
        >
          {value}
        </span>
        {sub && (
          <p className="text-[10px] text-slate-600 mt-2 font-mono tracking-wide">
            {sub}
          </p>
        )}
      </div>
    </div>
  );
}
