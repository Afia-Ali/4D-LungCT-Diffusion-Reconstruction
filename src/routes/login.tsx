import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { ShieldCheck, Lock, MonitorSmartphone } from "lucide-react";
import { Logo } from "@/components/Logo";
import { setUser } from "@/lib/lungsynth";

export const Route = createFileRoute("/login")({
  head: () => ({
    meta: [
      { title: "Sign in — LungSynth AI" },
      { name: "description", content: "Secure Google sign-in for radiologists and researchers using LungSynth AI 4D CT reconstruction." },
      { property: "og:title", content: "Sign in — LungSynth AI" },
      { property: "og:description", content: "Hospital-grade secure authentication for the LungSynth AI 4D CT platform." },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary_large_image" },
    ],
  }),
  component: Login,
});

const assurances = [
  { icon: ShieldCheck, label: "Secure Authentication" },
  { icon: Lock, label: "Hospital-grade privacy" },
  { icon: MonitorSmartphone, label: "Responsive design" },
];

function Login() {
  const navigate = useNavigate();

  const signIn = () => {
    setUser({ name: "Dr. Amara Reyes", email: "a.reyes@radiology.hospital.org", initials: "AR" });
    navigate({ to: "/dashboard" });
  };

  return (
    <div className="grid-medical flex min-h-screen items-center justify-center bg-background px-4 py-12">
      <div className="animate-fade-up w-full max-w-md">
        <div className="rounded-3xl border border-border bg-card p-8 shadow-[var(--shadow-lift)] sm:p-10">
          <div className="flex flex-col items-center text-center">
            <div className="text-primary">
              <Logo size={64} />
            </div>
            <h1 className="mt-6 text-2xl font-extrabold tracking-tight sm:text-[26px]">
              Welcome to LungSynth AI
            </h1>
            <p className="mt-3 text-sm leading-relaxed text-muted-foreground">
              Generate high-quality intermediate lung CT phases using AI diffusion models.
            </p>
          </div>

          <button
            onClick={signIn}
            className="mt-8 flex h-12 w-full items-center justify-center gap-3 rounded-xl border border-border bg-card text-sm font-semibold text-foreground shadow-[var(--shadow-soft)] transition-all hover:-translate-y-0.5 hover:bg-muted"
          >
            <svg width="18" height="18" viewBox="0 0 48 48" aria-hidden="true">
              <path fill="#EA4335" d="M24 9.5c3.5 0 6.6 1.2 9 3.6l6.7-6.7C35.6 2.6 30.2 0 24 0 14.6 0 6.5 5.4 2.6 13.2l7.8 6.1C12.3 13.2 17.7 9.5 24 9.5Z" />
              <path fill="#4285F4" d="M46.5 24.5c0-1.6-.1-3.2-.4-4.7H24v9h12.7c-.6 3-2.3 5.6-4.9 7.3l7.6 5.9c4.4-4.1 7.1-10.2 7.1-17.5Z" />
              <path fill="#FBBC05" d="M10.4 28.7a14.6 14.6 0 0 1 0-9.4l-7.8-6.1a24 24 0 0 0 0 21.6l7.8-6.1Z" />
              <path fill="#34A853" d="M24 48c6.5 0 11.9-2.1 15.9-5.8l-7.6-5.9c-2.1 1.4-4.9 2.3-8.3 2.3-6.3 0-11.7-3.7-13.6-9l-7.8 6.1C6.5 42.6 14.6 48 24 48Z" />
            </svg>
            Continue with Google
          </button>

          <div className="mt-8 flex flex-col gap-3 border-t border-border pt-6">
            {assurances.map((a) => (
              <div key={a.label} className="flex items-center gap-2.5 text-xs font-medium text-muted-foreground">
                <a.icon className="size-4 text-secondary" />
                {a.label}
              </div>
            ))}
          </div>
        </div>
        <p className="mt-6 text-center text-xs text-muted-foreground">
          Access restricted to authorised clinical and research personnel.
        </p>
      </div>
    </div>
  );
}
