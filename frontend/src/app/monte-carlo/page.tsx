"use client";

import React, { useEffect, useState, useMemo } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
} from "recharts";

interface MonteCarloDataPoint {
  hour: number;
  median: number;
  [key: string]: number; // path_0, path_1, ...
}

const API = "http://127.0.0.1:8000";

function MonteCarloTooltip({ active, payload }: any) {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload;
  const hour = d.hour;
  const median = d.median;

  return (
    <div className="bg-black/80 backdrop-blur-xl border border-white/10 rounded-xl px-5 py-3.5 shadow-2xl shadow-black/50">
      <p className="text-[10px] text-slate-500 font-mono uppercase tracking-wider mb-2">
        Forward Hour +{hour}
      </p>
      <div className="flex items-center justify-between gap-6">
        <span className="text-[11px] text-slate-400">Median Path</span>
        <span className="text-[13px] font-bold font-mono text-white">
          ${median.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
        </span>
      </div>
    </div>
  );
}

export default function MonteCarloPage() {
  const [data, setData] = useState<MonteCarloDataPoint[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    (async () => {
      try {
        const res = await fetch(`${API}/monte-carlo-visual`);
        if (!res.ok) throw new Error();
        const json = await res.json();
        setData(json);
      } catch (e) {
        console.error("Failed to load Monte Carlo data", e);
      } finally {
        setLoading(false);
      }
    })();
  }, []);

  // Determine path keys dynamically (e.g., path_0 to path_49)
  const pathKeys = useMemo(() => {
    if (!data.length) return [];
    return Object.keys(data[0]).filter((k) => k.startsWith("path_"));
  }, [data]);

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
            Risk Engine
          </h1>
          <p className="text-[11px] text-slate-600 mt-1 tracking-[0.15em] uppercase font-mono">
            Forward-Looking Monte Carlo Price Paths (24h)
          </p>
        </header>

        {loading ? (
          <div className="flex items-center justify-center h-[50vh]">
            <div className="w-10 h-10 rounded-full bg-white/[0.03] border border-white/[0.06] flex items-center justify-center">
              <div className="w-2.5 h-2.5 rounded-full bg-slate-600 animate-pulse" />
            </div>
          </div>
        ) : (
          <div className="rounded-2xl bg-white/[0.02] backdrop-blur-md border border-white/[0.06] p-6 shadow-xl shadow-black/20 flex flex-col h-[700px]">
            <div className="flex items-center justify-between mb-5 shrink-0">
              <div>
                <h2 className="text-[14px] font-semibold text-white tracking-tight">
                  Geometric Brownian Motion Simulation
                </h2>
                <p className="text-[11px] text-slate-600 mt-0.5 font-mono">
                  {pathKeys.length} Simulated Paths · Volatility Target: 5.0%
                </p>
              </div>
              <div className="flex items-center gap-5 text-[11px]">
                <span className="flex items-center gap-2 text-slate-400">
                  <span className="w-4 h-[2px] rounded-full bg-[#38bdf8] inline-block opacity-40" />
                  Simulation Path
                </span>
                <span className="flex items-center gap-2 text-white font-semibold">
                  <span
                    className="w-4 h-[2px] rounded-full bg-white inline-block"
                    style={{
                      backgroundImage:
                        "repeating-linear-gradient(90deg, #fff 0, #fff 4px, transparent 4px, transparent 8px)",
                    }}
                  />
                  Median Projection
                </span>
              </div>
            </div>

            <div className="w-full flex-1 min-h-0">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart
                  data={data}
                  margin={{ top: 20, right: 30, left: 10, bottom: 10 }}
                >
                  <XAxis
                    dataKey="hour"
                    stroke="transparent"
                    tick={{
                      fill: "#475569",
                      fontSize: 11,
                      fontFamily: "var(--font-geist-mono)",
                    }}
                    tickFormatter={(v) => `+${v}h`}
                    tickLine={false}
                    axisLine={false}
                    minTickGap={2}
                  />
                  <YAxis
                    stroke="transparent"
                    tick={{
                      fill: "#475569",
                      fontSize: 11,
                      fontFamily: "var(--font-geist-mono)",
                    }}
                    tickFormatter={(v) => `$${v.toLocaleString()}`}
                    tickLine={false}
                    axisLine={false}
                    domain={["auto", "auto"]}
                  />
                  <Tooltip
                    content={<MonteCarloTooltip />}
                    cursor={{ stroke: "rgba(255,255,255,0.06)", strokeWidth: 1 }}
                  />

                  {/* Render all 50 paths slightly transparent */}
                  {pathKeys.map((key) => (
                    <Line
                      key={key}
                      type="monotone"
                      dataKey={key}
                      stroke="#38bdf8"
                      strokeWidth={1}
                      strokeOpacity={0.1}
                      dot={false}
                      activeDot={false}
                      isAnimationActive={true}
                    />
                  ))}

                  {/* Render the median line boldly */}
                  <Line
                    type="monotone"
                    dataKey="median"
                    name="Median"
                    stroke="#ffffff"
                    strokeWidth={2}
                    strokeDasharray="6 6"
                    dot={false}
                    activeDot={{
                      r: 5,
                      fill: "#fff",
                      stroke: "rgba(255,255,255,0.3)",
                      strokeWidth: 6,
                    }}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
