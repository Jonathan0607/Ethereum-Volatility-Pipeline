import type { Metadata } from "next";
import { Inter, JetBrains_Mono } from "next/font/google";
import Link from "next/link";
import "./globals.css";

const inter = Inter({
  variable: "--font-sans",
  subsets: ["latin"],
  display: "swap",
});

const jetbrains = JetBrains_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
  display: "swap",
});

export const metadata: Metadata = {
  title: "Chronos — ETH Execution Engine",
  description: "Quantitative trading command center for Ethereum volatility strategies",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className={`${inter.variable} ${jetbrains.variable} font-sans bg-slate-950 text-slate-50 antialiased`}>
        {/* Persistent Navigation Bar */}
        <nav className="sticky top-0 z-50 w-full bg-slate-950/50 backdrop-blur-md border-b border-white/10">
          <div className="max-w-[1440px] mx-auto px-4 sm:px-6 lg:px-8">
            <div className="flex items-center justify-between h-16">
              <div className="flex items-center gap-8">
                {/* Logo */}
                <div className="flex items-center gap-3">
                  <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-blue-500 to-violet-600 flex items-center justify-center text-white font-bold text-sm shadow-lg shadow-indigo-500/25">
                    C
                  </div>
                  <span className="text-white font-bold tracking-tight">Chronos</span>
                </div>
                {/* Links */}
                <div className="hidden md:flex items-center gap-6">
                  <Link href="/" className="text-sm font-mono text-slate-400 hover:text-white transition-colors duration-200">
                    LiveOps
                  </Link>
                  <Link href="/backtest" className="text-sm font-mono text-slate-400 hover:text-white transition-colors duration-200">
                    Backtest
                  </Link>
                  <Link href="/monte-carlo" className="text-sm font-mono text-slate-400 hover:text-white transition-colors duration-200">
                    Risk Engine
                  </Link>
                </div>
              </div>
            </div>
          </div>
        </nav>
        {children}
      </body>
    </html>
  );
}
