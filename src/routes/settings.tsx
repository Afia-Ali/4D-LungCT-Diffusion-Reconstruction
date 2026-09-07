import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { useEffect, useState } from "react";
import { Pencil, LogOut, Trash2, Sun } from "lucide-react";
import { toast } from "sonner";
import { AppShell } from "@/components/AppShell";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import {
  clearHistory,
  clearUser,
  getPrefs,
  getUser,
  setPrefs,
  setUser,
  type Prefs,
  type StoredUser,
} from "@/lib/lungsynth";

export const Route = createFileRoute("/settings")({
  head: () => ({
    meta: [
      { title: "Settings — LungSynth AI" },
      { name: "description", content: "Manage your LungSynth AI profile, date and time preferences, notifications and local session history." },
      { property: "og:title", content: "Settings — LungSynth AI" },
      { property: "og:description", content: "Profile, preferences and account controls for the LungSynth AI platform." },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary_large_image" },
    ],
  }),
  component: SettingsPage,
});

function Section({ title, description, children }: { title: string; description: string; children: React.ReactNode }) {
  return (
    <section className="animate-fade-up rounded-2xl border border-border bg-card p-6 shadow-[var(--shadow-soft)] sm:p-7">
      <h2 className="text-lg font-bold">{title}</h2>
      <p className="mt-1 text-xs text-muted-foreground">{description}</p>
      <div className="mt-6 space-y-5">{children}</div>
    </section>
  );
}

function SettingsPage() {
  const navigate = useNavigate();
  const [user, setLocalUser] = useState<StoredUser | null>(null);
  const [prefs, setLocalPrefs] = useState<Prefs>({ dateFormat: "DD/MM/YYYY", timeFormat: "24h", notifications: true });
  const [editing, setEditing] = useState(false);
  const [name, setName] = useState("");

  useEffect(() => {
    const u = getUser() ?? { name: "Dr. Amara Reyes", email: "a.reyes@radiology.hospital.org", initials: "AR" };
    setLocalUser(u);
    setName(u.name);
    setLocalPrefs(getPrefs());
  }, []);

  const savePrefs = (next: Prefs) => {
    setLocalPrefs(next);
    setPrefs(next);
  };

  const saveName = () => {
    if (!user) return;
    const initials = name.split(" ").filter(Boolean).slice(-2).map((s) => s[0]?.toUpperCase() ?? "").join("");
    const next = { ...user, name, initials: initials || user.initials };
    setLocalUser(next);
    setUser(next);
    setEditing(false);
    toast.success("Profile name updated");
  };

  return (
    <AppShell>
      <div className="animate-fade-up">
        <h1 className="text-3xl font-extrabold tracking-tight sm:text-4xl">Settings</h1>
        <p className="mt-2.5 text-sm text-muted-foreground sm:text-base">Manage your account and workspace preferences.</p>
      </div>

      <div className="mt-8 grid gap-6 lg:grid-cols-2">
        <Section title="Profile" description="Your clinical account details.">
          <div className="flex items-center gap-4">
            <span className="flex size-14 items-center justify-center rounded-full bg-primary text-base font-bold text-primary-foreground">
              {user?.initials}
            </span>
            <div className="min-w-0 flex-1">
              <Label className="text-xs text-muted-foreground">User name</Label>
              {editing ? (
                <div className="mt-1.5 flex gap-2">
                  <Input value={name} onChange={(e) => setName(e.target.value)} className="h-10 rounded-xl" />
                  <Button className="rounded-xl" onClick={saveName}>Save</Button>
                </div>
              ) : (
                <p className="mt-1 truncate text-sm font-semibold">{user?.name}</p>
              )}
            </div>
            {!editing ? (
              <Button variant="outline" className="rounded-xl" onClick={() => setEditing(true)}>
                <Pencil className="size-4" /> Edit Name
              </Button>
            ) : null}
          </div>
          <div>
            <Label className="text-xs text-muted-foreground">Email (read-only)</Label>
            <Input readOnly value={user?.email ?? ""} className="mt-1.5 h-11 rounded-xl bg-muted" />
          </div>
        </Section>

        <Section title="Preferences" description="Display and notification settings.">
          <div className="grid gap-4 sm:grid-cols-2">
            <div>
              <Label className="text-xs text-muted-foreground">Preferred date format</Label>
              <Select value={prefs.dateFormat} onValueChange={(v) => savePrefs({ ...prefs, dateFormat: v })}>
                <SelectTrigger className="mt-1.5 h-11 w-full rounded-xl"><SelectValue /></SelectTrigger>
                <SelectContent>
                  <SelectItem value="DD/MM/YYYY">DD/MM/YYYY</SelectItem>
                  <SelectItem value="MM/DD/YYYY">MM/DD/YYYY</SelectItem>
                  <SelectItem value="YYYY-MM-DD">YYYY-MM-DD</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div>
              <Label className="text-xs text-muted-foreground">Preferred time format</Label>
              <Select value={prefs.timeFormat} onValueChange={(v) => savePrefs({ ...prefs, timeFormat: v })}>
                <SelectTrigger className="mt-1.5 h-11 w-full rounded-xl"><SelectValue /></SelectTrigger>
                <SelectContent>
                  <SelectItem value="24h">24-hour</SelectItem>
                  <SelectItem value="12h">12-hour</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>
          <div className="flex items-center justify-between rounded-xl border border-border bg-muted/50 px-4 py-3">
            <span className="flex items-center gap-2 text-sm font-medium"><Sun className="size-4 text-secondary" /> Theme</span>
            <span className="text-xs font-semibold text-muted-foreground">Light (only for now)</span>
          </div>
          <div className="flex items-center justify-between rounded-xl border border-border px-4 py-3">
            <div>
              <p className="text-sm font-medium">Notifications</p>
              <p className="text-xs text-muted-foreground">Alert me when a reconstruction completes.</p>
            </div>
            <Switch
              checked={prefs.notifications}
              onCheckedChange={(v) => savePrefs({ ...prefs, notifications: v })}
            />
          </div>
        </Section>

        <Section title="Account" description="Session controls.">
          <Button
            variant="outline"
            className="h-11 w-full rounded-xl"
            onClick={() => {
              clearUser();
              navigate({ to: "/login" });
            }}
          >
            <LogOut className="size-4" /> Logout
          </Button>
        </Section>

        <Section title="Danger Zone" description="Irreversible local actions.">
          <div className="flex flex-col gap-3 rounded-xl border border-destructive/30 bg-destructive/5 p-4 sm:flex-row sm:items-center sm:justify-between">
            <div>
              <p className="text-sm font-semibold text-destructive">Clear local history</p>
              <p className="text-xs text-muted-foreground">Removes all stored session records from this device.</p>
            </div>
            <Button
              variant="destructive"
              className="rounded-xl"
              onClick={() => {
                clearHistory();
                toast.success("Local history cleared");
              }}
            >
              <Trash2 className="size-4" /> Clear
            </Button>
          </div>
        </Section>
      </div>
    </AppShell>
  );
}
