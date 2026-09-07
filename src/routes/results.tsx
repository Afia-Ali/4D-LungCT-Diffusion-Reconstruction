import { createFileRoute } from "@tanstack/react-router";
import { useMemo, useState } from "react";
import { Download, FileDown, Maximize2, ZoomIn, ZoomOut, Move, Expand } from "lucide-react";
import { AppShell } from "@/components/AppShell";
import { CtPreview } from "@/components/CtPreview";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogTitle } from "@/components/ui/dialog";
import { buildPhases, MODEL_VERSION, type PhaseResult } from "@/lib/lungsynth";

export const Route = createFileRoute("/results")({
  head: () => ({
    meta: [
      { title: "Generated CT Phases — LungSynth AI" },
      { name: "description", content: "Review AI-generated intermediate lung CT breathing phases T10 through T80 with confidence scores and metadata." },
      { property: "og:title", content: "Generated CT Phases — LungSynth AI" },
      { property: "og:description", content: "Inspect, zoom and export AI-reconstructed intermediate 4D lung CT phases." },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary_large_image" },
    ],
  }),
  component: Results,
});

function Results() {
  const phases = useMemo(() => buildPhases(1_760_000_000_000), []);
  const [open, setOpen] = useState<PhaseResult | null>(null);
  const [zoom, setZoom] = useState(1);

  return (
    <AppShell>
      <div className="animate-fade-up flex flex-col gap-5 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <h1 className="text-3xl font-extrabold tracking-tight sm:text-4xl">
            Generated Intermediate Lung CT Phases
          </h1>
          <p className="mt-2.5 text-sm text-muted-foreground sm:text-base">
            AI-generated breathing phases between T00 and T50.
          </p>
        </div>
        <div className="flex shrink-0 gap-2.5">
          <Button variant="outline" className="rounded-xl">
            <FileDown className="size-4" /> Export Results
          </Button>
          <Button className="rounded-xl">
            <Download className="size-4" /> Download All
          </Button>
        </div>
      </div>

      <div className="mt-8 grid gap-5 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
        {phases.map((p, i) => (
          <button
            key={p.id}
            onClick={() => {
              setZoom(1);
              setOpen(p);
            }}
            style={{ animationDelay: `${i * 0.05}s` }}
            className="animate-fade-up group rounded-2xl border border-border bg-card p-4 text-left shadow-[var(--shadow-soft)] transition-all hover:-translate-y-1 hover:shadow-[var(--shadow-lift)]"
          >
            <div className="flex items-center justify-between">
              <span className="rounded-lg bg-accent px-2.5 py-1 text-xs font-bold text-accent-foreground">
                {p.label}
              </span>
              <Expand className="size-4 text-muted-foreground transition-colors group-hover:text-primary" />
            </div>
            <CtPreview className="mt-3 aspect-square w-full" label={p.label} />
            <dl className="mt-4 space-y-1.5 text-xs">
              <div className="flex justify-between">
                <dt className="text-muted-foreground">Confidence</dt>
                <dd className="font-semibold text-success">{(p.confidence * 100).toFixed(1)}%</dd>
              </div>
              <div className="flex justify-between">
                <dt className="text-muted-foreground">Generation time</dt>
                <dd className="font-semibold">{(p.generationMs / 1000).toFixed(2)}s</dd>
              </div>
            </dl>
          </button>
        ))}
      </div>

      <Dialog open={Boolean(open)} onOpenChange={(o) => !o && setOpen(null)}>
        <DialogContent className="max-w-5xl gap-0 overflow-hidden rounded-2xl p-0">
          <DialogTitle className="sr-only">Phase {open?.label} viewer</DialogTitle>
          {open ? (
            <div className="grid lg:grid-cols-[1fr_18rem]">
              <div className="relative bg-foreground/95 p-4">
                <div className="relative aspect-square w-full overflow-hidden rounded-xl">
                  <div
                    className="size-full origin-center transition-transform duration-200"
                    style={{ transform: `scale(${zoom})` }}
                  >
                    <CtPreview className="size-full" label={open.label} />
                  </div>
                </div>
                <div className="mt-3 flex flex-wrap items-center gap-2">
                  <Button size="sm" variant="secondary" className="rounded-lg" onClick={() => setZoom((z) => Math.min(3, z + 0.25))}>
                    <ZoomIn className="size-4" /> Zoom in
                  </Button>
                  <Button size="sm" variant="secondary" className="rounded-lg" onClick={() => setZoom((z) => Math.max(1, z - 0.25))}>
                    <ZoomOut className="size-4" /> Zoom out
                  </Button>
                  <Button size="sm" variant="secondary" className="rounded-lg">
                    <Move className="size-4" /> Pan
                  </Button>
                  <Button size="sm" variant="secondary" className="rounded-lg">
                    <Maximize2 className="size-4" /> Full screen
                  </Button>
                  <Button size="sm" className="ml-auto rounded-lg">
                    <Download className="size-4" /> Download Image
                  </Button>
                </div>
              </div>

              <aside className="border-t border-border bg-card p-6 lg:border-l lg:border-t-0">
                <h2 className="text-lg font-bold">Phase {open.label}</h2>
                <p className="mt-1 text-xs text-muted-foreground">Interpolated breathing phase</p>
                <dl className="mt-6 space-y-4 text-sm">
                  {[
                    ["Image resolution", open.resolution],
                    ["Generation timestamp", new Date(open.timestamp).toLocaleString()],
                    ["AI model version", MODEL_VERSION],
                    ["Processing duration", `${(open.generationMs / 1000).toFixed(2)}s`],
                    ["Confidence score", `${(open.confidence * 100).toFixed(1)}%`],
                  ].map(([k, v]) => (
                    <div key={k}>
                      <dt className="text-xs font-medium text-muted-foreground">{k}</dt>
                      <dd className="mt-0.5 font-semibold">{v}</dd>
                    </div>
                  ))}
                </dl>
              </aside>
            </div>
          ) : null}
        </DialogContent>
      </Dialog>
    </AppShell>
  );
}
