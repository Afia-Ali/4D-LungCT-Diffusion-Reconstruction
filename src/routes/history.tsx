import { createFileRoute, Link } from "@tanstack/react-router";
import { useEffect, useMemo, useState } from "react";
import { Search, CheckCircle2, Clock, Layers, CalendarDays } from "lucide-react";
import { AppShell } from "@/components/AppShell";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Skeleton } from "@/components/ui/skeleton";
import { getHistory, type HistoryEntry } from "@/lib/lungsynth";

export const Route = createFileRoute("/history")({
  head: () => ({
    meta: [
      { title: "Session History — LungSynth AI" },
      { name: "description", content: "Browse previous 4D lung CT reconstruction sessions with timestamps, phase counts and processing times." },
      { property: "og:title", content: "Session History — LungSynth AI" },
      { property: "og:description", content: "Timeline of past AI CT phase generation sessions." },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary_large_image" },
    ],
  }),
  component: HistoryPage,
});

const FILTERS = ["Today", "Last Week", "Last Month", "All"] as const;

function HistoryPage() {
  const [items, setItems] = useState<HistoryEntry[] | null>(null);
  const [query, setQuery] = useState("");
  const [filter, setFilter] = useState<(typeof FILTERS)[number]>("All");

  useEffect(() => {
    const t = setTimeout(() => setItems(getHistory()), 500);
    return () => clearTimeout(t);
  }, []);

  const shown = useMemo(() => {
    if (!items) return [];
    const now = Date.now();
    const windows: Record<string, number> = {
      Today: 864e5,
      "Last Week": 7 * 864e5,
      "Last Month": 30 * 864e5,
      All: Number.POSITIVE_INFINITY,
    };
    return items
      .filter((i) => now - new Date(i.createdAt).getTime() <= (windows[filter] ?? Infinity))
      .filter((i) => i.sessionId.toLowerCase().includes(query.toLowerCase()))
      .sort((a, b) => +new Date(b.createdAt) - +new Date(a.createdAt));
  }, [items, filter, query]);

  return (
    <AppShell>
      <div className="animate-fade-up">
        <h1 className="text-3xl font-extrabold tracking-tight sm:text-4xl">Session History</h1>
        <p className="mt-2.5 text-sm text-muted-foreground sm:text-base">
          Previous reconstruction sessions, newest first.
        </p>
      </div>

      <div className="animate-fade-up mt-7 flex flex-col gap-3 sm:flex-row sm:items-center">
        <div className="relative flex-1">
          <Search className="absolute left-3.5 top-1/2 size-4 -translate-y-1/2 text-muted-foreground" />
          <Input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search by session ID"
            className="h-11 rounded-xl bg-card pl-10"
          />
        </div>
        <div className="flex gap-1.5 rounded-xl border border-border bg-card p-1">
          {FILTERS.map((f) => (
            <button
              key={f}
              onClick={() => setFilter(f)}
              className={`rounded-lg px-3 py-2 text-xs font-semibold transition-colors ${
                filter === f ? "bg-primary text-primary-foreground" : "text-muted-foreground hover:bg-muted"
              }`}
            >
              {f}
            </button>
          ))}
        </div>
      </div>

      <div className="relative mt-8 pl-6">
        <span className="absolute left-[7px] top-2 bottom-2 w-px bg-border" />
        <div className="flex flex-col gap-5">
          {!items
            ? [0, 1, 2].map((i) => <Skeleton key={i} className="h-32 rounded-2xl" />)
            : shown.map((h, i) => {
                const d = new Date(h.createdAt);
                return (
                  <div key={h.id} className="relative animate-fade-up" style={{ animationDelay: `${i * 0.06}s` }}>
                    <span className="absolute -left-[22px] top-7 size-3 rounded-full border-2 border-card bg-secondary ring-4 ring-background" />
                    <div className="rounded-2xl border border-border bg-card p-5 shadow-[var(--shadow-soft)] transition-all hover:-translate-y-0.5 hover:shadow-[var(--shadow-lift)] sm:p-6">
                      <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
                        <div>
                          <div className="flex flex-wrap items-center gap-2">
                            <span className="font-mono text-sm font-bold">{h.sessionId}</span>
                            <span className="inline-flex items-center gap-1 rounded-full bg-success/10 px-2.5 py-1 text-[11px] font-semibold text-success">
                              <CheckCircle2 className="size-3" /> {h.status}
                            </span>
                          </div>
                          <div className="mt-3 flex flex-wrap gap-x-5 gap-y-2 text-xs text-muted-foreground">
                            <span className="inline-flex items-center gap-1.5">
                              <CalendarDays className="size-3.5" /> {d.toLocaleDateString()} · {d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
                            </span>
                            <span className="inline-flex items-center gap-1.5">
                              <Layers className="size-3.5" /> {h.phases} phases generated
                            </span>
                            <span className="inline-flex items-center gap-1.5">
                              <Clock className="size-3.5" /> {h.processingSeconds}s processing
                            </span>
                          </div>
                        </div>
                        <Button asChild variant="outline" className="rounded-xl">
                          <Link to="/results">View Results</Link>
                        </Button>
                      </div>
                    </div>
                  </div>
                );
              })}
          {items && shown.length === 0 ? (
            <p className="rounded-2xl border border-dashed border-border bg-card p-10 text-center text-sm text-muted-foreground">
              No sessions match your search.
            </p>
          ) : null}
        </div>
      </div>
    </AppShell>
  );
}
