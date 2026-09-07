export function CtPreview({ label, className = "" }: { label?: string; className?: string }) {
  return (
    <div className={`ct-surface relative overflow-hidden rounded-xl ${className}`}>
      <div className="absolute inset-0 opacity-40 mix-blend-screen [background:repeating-linear-gradient(0deg,transparent_0_3px,rgba(255,255,255,0.04)_3px_4px)]" />
      <div className="absolute inset-x-0 h-1/4 animate-scan-sweep bg-[linear-gradient(to_bottom,transparent,rgba(255,255,255,0.10),transparent)]" />
      {label ? (
        <span className="absolute bottom-2 left-2.5 rounded-md bg-black/40 px-1.5 py-0.5 font-mono text-[10px] tracking-wider text-white/80">
          {label}
        </span>
      ) : null}
    </div>
  );
}
