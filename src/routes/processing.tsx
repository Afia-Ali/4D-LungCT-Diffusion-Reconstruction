import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { useEffect, useState } from "react";
import { Check, Loader2 } from "lucide-react";
import { AppShell } from "@/components/AppShell";
import { addHistory, newSessionId } from "@/lib/lungsynth";

export const Route = createFileRoute("/processing")({
  head: () => ({
    meta: [
      { title: "Processing — LungSynth AI" },
      { name: "description", content: "The phase-conditioned diffusion model is reconstructing the intermediate lung CT breathing cycle." },
      { property: "og:title", content: "Processing — LungSynth AI" },
      { property: "og:description", content: "Live progress for AI reconstruction of intermediate 4D lung CT phases." },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary_large_image" },
    ],
  }),
  component: Processing,
});

const STAGES = [
  "Uploading scans",
  "Preprocessing",
  "Running Diffusion Model",
  "Generating CT Phases",
  "Finalizing Results",
];

const TOTAL_MS = 14000;

function Processing() {
  const navigate = useNavigate();
  const [progress, setProgress] = useState(0);

  useEffect(() => {
    const start = Date.now();
    const id = setInterval(() => {
      const p = Math.min(100, ((Date.now() - start) / TOTAL_MS) * 100);
      setProgress(p);
      if (p >= 100) {
        clearInterval(id);
        addHistory({
          id: crypto.randomUUID(),
          sessionId: newSessionId(),
          createdAt: new Date().toISOString(),
          phases: 7,
          processingSeconds: Math.round(TOTAL_MS / 1000),
          status: "Completed",
        });
        navigate({ to: "/results" });
      }
    }, 120);
    return () => clearInterval(id);
  }, [navigate]);

  const activeStage = Math.min(STAGES.length - 1, Math.floor((progress / 100) * STAGES.length));
  const remaining = Math.max(1, Math.round(((100 - progress) / 100) * (TOTAL_MS / 1000)));

  return (
    <AppShell>
      <div className="mx-auto flex max-w-2xl flex-col items-center text-center">
        <div className="animate-fade-up relative size-44">
          <div className="absolute inset-0 animate-ct-spin rounded-full border-2 border-dashed border-primary/25" />
          <div className="absolute inset-3 animate-ct-spin rounded-full border-2 border-secondary/30 [animation-direction:reverse] [animation-duration:5s]" />
          <div className="absolute inset-8 overflow-hidden rounded-full">
            <div className="ct-surface size-full" />
            <div className="absolute inset-x-0 h-1/3 animate-scan-sweep bg-[linear-gradient(to_bottom,transparent,rgba(255,255,255,0.16),transparent)]" />
          </div>
          <div className="absolute inset-0 animate-ct-spin [animation-duration:2.4s]">
            <span className="absolute left-1/2 top-0 size-2.5 -translate-x-1/2 rounded-full bg-secondary" />
          </div>
        </div>

        <h1 className="animate-fade-up mt-10 text-2xl font-extrabold tracking-tight sm:text-3xl">
          Generating Intermediate CT Phases
        </h1>
        <p className="animate-fade-up mt-2.5 text-sm text-muted-foreground">
          Please wait while the AI reconstructs the breathing cycle.
        </p>

        <div className="mt-10 w-full rounded-2xl border border-border bg-card p-6 text-left shadow-[var(--shadow-soft)]">
          <ul className="flex flex-col gap-4">
            {STAGES.map((s, i) => {
              const done = i < activeStage;
              const active = i === activeStage;
              return (
                <li key={s} className="flex items-center gap-3">
                  <span
                    className={`flex size-7 items-center justify-center rounded-full border transition-colors ${
                      done
                        ? "border-success bg-success text-success-foreground"
                        : active
                          ? "border-primary bg-accent text-primary"
                          : "border-border bg-muted text-muted-foreground"
                    }`}
                  >
                    {done ? (
                      <Check className="size-4" />
                    ) : active ? (
                      <Loader2 className="size-4 animate-spin" />
                    ) : (
                      <span className="text-[11px] font-bold">{i + 1}</span>
                    )}
                  </span>
                  <div className="flex-1">
                    <p className={`text-sm font-semibold ${done || active ? "text-foreground" : "text-muted-foreground"}`}>
                      Stage {i + 1} · {s}
                    </p>
                  </div>
                </li>
              );
            })}
          </ul>

          <div className="mt-6">
            <div className="h-2 w-full overflow-hidden rounded-full bg-muted">
              <div
                className="h-full rounded-full bg-primary transition-[width] duration-200 ease-linear"
                style={{ width: `${progress}%` }}
              />
            </div>
            <div className="mt-2.5 flex justify-between text-xs font-medium text-muted-foreground">
              <span>{Math.round(progress)}% complete</span>
              <span>Estimated remaining: {remaining}s</span>
            </div>
          </div>
        </div>
      </div>
    </AppShell>
  );
}
