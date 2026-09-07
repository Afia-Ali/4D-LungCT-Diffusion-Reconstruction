import { Link, useNavigate, useRouterState } from "@tanstack/react-router";
import { History, Settings, Menu, LogOut } from "lucide-react";
import { useEffect, useState } from "react";
import { LogoWordmark } from "./Logo";
import { clearUser, getUser, type StoredUser } from "@/lib/lungsynth";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Sheet, SheetContent, SheetTrigger } from "@/components/ui/sheet";

export function AppNav() {
  const [user, setUserState] = useState<StoredUser | null>(null);
  const navigate = useNavigate();
  const path = useRouterState({ select: (s) => s.location.pathname });

  useEffect(() => {
    setUserState(getUser());
  }, []);

  const logout = () => {
    clearUser();
    navigate({ to: "/login" });
  };

  const links = [
    { to: "/dashboard", label: "Generate" },
    { to: "/history", label: "History" },
    { to: "/settings", label: "Settings" },
  ] as const;

  return (
    <header className="glass-panel fixed inset-x-0 top-0 z-50 border-b">
      <div className="mx-auto flex h-16 max-w-7xl items-center justify-between px-4 sm:px-6">
        <LogoWordmark />

        <nav className="hidden items-center gap-1 md:flex">
          {links.map((l) => (
            <Link
              key={l.to}
              to={l.to}
              className={`rounded-lg px-3 py-2 text-sm font-medium transition-colors ${
                path === l.to
                  ? "bg-accent text-accent-foreground"
                  : "text-muted-foreground hover:bg-muted hover:text-foreground"
              }`}
            >
              {l.label}
            </Link>
          ))}
        </nav>

        <div className="flex items-center gap-1.5">
          <Button asChild variant="ghost" size="icon" className="hidden rounded-xl sm:inline-flex">
            <Link to="/history" aria-label="History">
              <History className="size-[18px]" />
            </Link>
          </Button>
          <Button asChild variant="ghost" size="icon" className="hidden rounded-xl sm:inline-flex">
            <Link to="/settings" aria-label="Settings">
              <Settings className="size-[18px]" />
            </Link>
          </Button>

          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <button
                aria-label="Account"
                className="ml-1 flex size-9 items-center justify-center rounded-full bg-primary text-xs font-bold text-primary-foreground transition-transform hover:scale-105"
              >
                {user?.initials ?? "RA"}
              </button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end" className="w-56 rounded-xl">
              <div className="px-2 py-1.5">
                <p className="text-sm font-semibold">{user?.name ?? "Dr. Radiologist"}</p>
                <p className="truncate text-xs text-muted-foreground">
                  {user?.email ?? "radiology@hospital.org"}
                </p>
              </div>
              <DropdownMenuSeparator />
              <DropdownMenuItem onClick={() => navigate({ to: "/settings" })}>
                <Settings className="size-4" /> Settings
              </DropdownMenuItem>
              <DropdownMenuItem onClick={logout}>
                <LogOut className="size-4" /> Log out
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>

          <Sheet>
            <SheetTrigger asChild>
              <Button variant="ghost" size="icon" className="rounded-xl md:hidden" aria-label="Menu">
                <Menu className="size-[18px]" />
              </Button>
            </SheetTrigger>
            <SheetContent side="right" className="w-64 p-6">
              <nav className="mt-8 flex flex-col gap-1">
                {links.map((l) => (
                  <Link
                    key={l.to}
                    to={l.to}
                    className="rounded-lg px-3 py-2.5 text-sm font-medium text-foreground hover:bg-muted"
                  >
                    {l.label}
                  </Link>
                ))}
              </nav>
            </SheetContent>
          </Sheet>
        </div>
      </div>
    </header>
  );
}

export function AppShell({ children }: { children: React.ReactNode }) {
  return (
    <div className="min-h-screen bg-background">
      <AppNav />
      <main className="mx-auto max-w-7xl px-4 pt-28 pb-20 sm:px-6">{children}</main>
    </div>
  );
}
