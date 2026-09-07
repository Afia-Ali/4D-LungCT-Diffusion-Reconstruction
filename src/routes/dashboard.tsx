import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { useRef, useState } from "react";
import { UploadCloud, CheckCircle2, FileText, X, Sparkles } from "lucide-react";
import { AppShell } from "@/components/AppShell";
import { CtPreview } from "@/components/CtPreview";
import { Button } from "@/components/ui/button";
import { formatBytes } from "@/lib/lungsynth";

export const Route = createFileRoute("/dashboard")({
  head: () => ({
    meta: [
      { title: "Generate CT Phases — LungSynth AI" },
      { name: "description", content: "Upload T00 and T50 boundary lung CT scans to generate intermediate 4D breathing phases with AI." },
      { property: "og:title", content: "Generate CT Phases — LungSynth AI" },
      { property: "og:description", content: "Upload boundary phase CT scans and reconstruct the full breathing cycle with a diffusion model." },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary_large_image" },
    ],
  }),
  component: Dashboard,
});

type Picked = { name: string; size: number } | null;

const ACCEPT = ".dcm,.nii,.nii.gz,.mha,.mhd,.nrrd";

function UploadCard({
  title,
  hint,
  value,
  onChange,
  delay,
}: {
  title: string;
  hint: string;
  value: Picked;
  onChange: (v: Picked) => void;
  delay: string;
}) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [over, setOver] = useState(false);

  const take = (files: FileList | null) => {
    const f = files?.[0];
    if (f) onChange({ name: f.name, size: f.size });
  };

  return (
    <div
      className="animate-fade-up rounded-2xl border border-border bg-card p-6 shadow-[var(--shadow-soft)] transition-shadow hover:shadow-[var(--shadow-lift)]"
      style={{ animationDelay: delay }}
    >
      <div className="flex items-center justify-between">
        <h3 className="text-base font-bold">{title}</h3>
        <span className="rounded-full bg-muted px-2.5 py-1 text-[11px] font-semibold text-muted-foreground">
          {hint}
        </span>
      </div>

      {value ? (
        <div className="mt-5">
          <div className="flex items-center gap-4">
            <CtPreview className="size-20 shrink-0" />
            <div className="min-w-0 flex-1">
              <div className="flex items-center gap-2">
                <CheckCircle2 className="size-4 shrink-0 text-success" />
                <p className="truncate text-sm font-semibold">{value.name}</p>
              </div>
              <p className="mt-1 flex items-center gap-1.5 text-xs text-muted-foreground">
                <FileText className="size-3.5" /> {formatBytes(value.size)} · Ready
              </p>
            </div>
            <Button variant="ghost" size="icon" className="rounded-lg" onClick={() => onChange(null)} aria-label="Remove file">
              <X className="size-4" />
            </Button>
          </div>
        </div>
      ) : (
        <div
          onDragOver={(e) => {
            e.preventDefault();
            setOver(true);
          }}
          onDragLeave={() => setOver(false)}
          onDrop={(e) => {
            e.preventDefault();
            setOver(false);
            take(e.dataTransfer.files);
          }}
          onClick={() => inputRef.current?.click()}
          className={`mt-5 flex cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed px-6 py-10 text-center transition-all ${
            over ? "border-secondary bg-accent" : "border-border hover:border-primary/40 hover:bg-muted/60"
          }`}
        >
          <UploadCloud className="size-9 text-primary" />
          <p className="mt-4 text-sm font-semibold">Drag &amp; Drop</p>
          <p className="mt-1 text-xs text-muted-foreground">
            or <span className="font-semibold text-primary underline underline-offset-2">Browse Files</span>
          </p>
          <p className="mt-4 font-mono text-[11px] text-muted-foreground">Accepted: DICOM (.dcm), NIfTI (.nii/.nii.gz), MetaImage (.mha/.mhd), NRRD (.nrrd)</p>
          <input
            ref={inputRef}
            type="file"
            accept={ACCEPT}
            className="hidden"
            onChange={(e) => take(e.target.files)}
          />
        </div>
      )}
    </div>
  );
}

function Dashboard() {
  const [t00, setT00] = useState<Picked>(null);
  const [t50, setT50] = useState<Picked>(null);
  const navigate = useNavigate();
  const ready = Boolean(t00 && t50);

  return (
    <AppShell>
      <div className="animate-fade-up">
        <h1 className="text-3xl font-extrabold tracking-tight sm:text-4xl">Generate Intermediate CT Phases</h1>
        <p className="mt-2.5 text-sm text-muted-foreground sm:text-base">
          Upload the required boundary phase CT scans.
        </p>
      </div>

      <div className="mt-8 grid gap-6 lg:grid-cols-2">
        <UploadCard title="Upload T00 Scan" hint="End inhalation" value={t00} onChange={setT00} delay="0.05s" />
        <UploadCard title="Upload T50 Scan" hint="End exhalation" value={t50} onChange={setT50} delay="0.12s" />
      </div>

      <div className="animate-fade-up mt-10 flex flex-col items-center gap-3 [animation-delay:0.2s]">
        <Button
          size="lg"
          disabled={!ready}
          onClick={() => navigate({ to: "/processing" })}
          className="h-14 w-full rounded-2xl px-10 text-base font-bold shadow-[var(--shadow-soft)] transition-all hover:-translate-y-0.5 hover:shadow-[var(--shadow-lift)] sm:w-auto"
        >
          <Sparkles className="size-5" />
          Generate CT Phases
        </Button>
        <p className="text-xs text-muted-foreground">
          {ready ? "7 intermediate phases will be reconstructed (T10–T80)." : "Upload both boundary scans to enable generation."}
        </p>
      </div>
    </AppShell>
  );
}
