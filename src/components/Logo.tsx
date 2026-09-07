import { Link } from "@tanstack/react-router";

export function Logo({ size = 36 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 48 48" fill="none" aria-hidden="true">
      <rect width="48" height="48" rx="14" fill="currentColor" opacity="0.08" />
      <path
        d="M24 12v10m0 0c-2.5-4-6-6-9-5.2-2.6.7-3.4 3.4-3 7.2.6 5.6 2.6 11 5.4 12.4 2.6 1.3 4.6-.7 5.2-4.2.3-1.9.4-6.4.4-10.2Zm0 0c2.5-4 6-6 9-5.2 2.6.7 3.4 3.4 3 7.2-.6 5.6-2.6 11-5.4 12.4-2.6 1.3-4.6-.7-5.2-4.2-.3-1.9-.4-6.4-.4-10.2Z"
        stroke="currentColor"
        strokeWidth="2.1"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

export function LogoWordmark({ size = 32 }: { size?: number }) {
  return (
    <Link to="/dashboard" className="flex items-center gap-2.5 text-primary">
      <Logo size={size} />
      <span className="text-[17px] font-extrabold tracking-tight text-foreground">
        LungSynth <span className="text-primary">AI</span>
      </span>
    </Link>
  );
}
