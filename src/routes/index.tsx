import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { useEffect } from "react";
import { Logo } from "@/components/Logo";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "LungSynth AI — 4D Lung CT Phase Reconstruction" },
      {
        name: "description",
        content:
          "LungSynth AI generates intermediate 4D lung CT phases from T00 and T50 boundary scans using a phase-conditioned diffusion model.",
      },
      { property: "og:title", content: "LungSynth AI — 4D Lung CT Phase Reconstruction" },
      {
        property: "og:description",
        content: "AI-powered reconstruction of intermediate lung CT breathing phases for radiologists and researchers.",
      },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary_large_image" },
    ],
  }),
  component: Splash,
});

function Splash() {
  const navigate = useNavigate();

  useEffect(() => {
    const t = setTimeout(() => navigate({ to: "/login" }), 2200);
    return () => clearTimeout(t);
  }, [navigate]);

  return (
    <div className="grid-medical relative flex min-h-screen items-center justify-center bg-background px-6">
      <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(ellipse_at_center,transparent_10%,var(--color-background)_78%)]" />
      <div className="relative flex flex-col items-center text-center">
        <div className="animate-fade-up text-primary [animation-duration:1.1s]">
          <Logo size={92} />
        </div>
        <h1 className="animate-fade-up mt-7 text-4xl font-extrabold tracking-tight [animation-delay:0.25s] sm:text-5xl">
          LungSynth <span className="text-primary">AI</span>
        </h1>
        <p className="animate-fade-up mt-3 text-sm font-medium text-muted-foreground [animation-delay:0.5s] sm:text-base">
          AI-powered 4D CT Reconstruction
        </p>
        <div className="animate-fade-up mt-10 flex items-center gap-2 [animation-delay:0.8s]">
          <span className="size-1.5 animate-soft-pulse rounded-full bg-primary" />
          <span className="size-1.5 animate-soft-pulse rounded-full bg-primary [animation-delay:0.2s]" />
          <span className="size-1.5 animate-soft-pulse rounded-full bg-primary [animation-delay:0.4s]" />
          <span className="ml-2 text-xs font-medium tracking-wide text-muted-foreground">Loading...</span>
        </div>
      </div>
    </div>
  );
}
