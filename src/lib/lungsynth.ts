export type PhaseResult = {
  id: string;
  label: string;
  confidence: number;
  generationMs: number;
  resolution: string;
  timestamp: string;
};

export type HistoryEntry = {
  id: string;
  sessionId: string;
  createdAt: string;
  phases: number;
  processingSeconds: number;
  status: "Completed";
};

export type StoredUser = {
  name: string;
  email: string;
  initials: string;
};

const USER_KEY = "lungsynth.user";
const HISTORY_KEY = "lungsynth.history";
const PREFS_KEY = "lungsynth.prefs";

export const PHASE_LABELS = ["T10", "T20", "T30", "T40", "T60", "T70", "T80"];

export const MODEL_VERSION = "PhaseDiff v2.4.1";

export function buildPhases(seed = Date.now()): PhaseResult[] {
  return PHASE_LABELS.map((label, i) => ({
    id: label,
    label,
    confidence: 0.93 + (((seed / 1000 + i * 37) % 60) / 1000),
    generationMs: 820 + ((i * 137) % 460),
    resolution: "512 × 512 × 128",
    timestamp: new Date(seed + i * 1200).toISOString(),
  }));
}

function read<T>(key: string, fallback: T): T {
  if (typeof window === "undefined") return fallback;
  try {
    const raw = window.localStorage.getItem(key);
    return raw ? (JSON.parse(raw) as T) : fallback;
  } catch {
    return fallback;
  }
}

function write(key: string, value: unknown) {
  if (typeof window === "undefined") return;
  window.localStorage.setItem(key, JSON.stringify(value));
}

export const getUser = () => read<StoredUser | null>(USER_KEY, null);
export const setUser = (u: StoredUser) => write(USER_KEY, u);
export const clearUser = () => window.localStorage.removeItem(USER_KEY);

export const getHistory = () => read<HistoryEntry[]>(HISTORY_KEY, seedHistory());
export const addHistory = (entry: HistoryEntry) => write(HISTORY_KEY, [entry, ...getHistory()]);
export const clearHistory = () => write(HISTORY_KEY, []);

export type Prefs = {
  dateFormat: string;
  timeFormat: string;
  notifications: boolean;
};

export const getPrefs = () =>
  read<Prefs>(PREFS_KEY, { dateFormat: "DD/MM/YYYY", timeFormat: "24h", notifications: true });
export const setPrefs = (p: Prefs) => write(PREFS_KEY, p);

export function newSessionId() {
  return `LS-${Math.random().toString(36).slice(2, 7).toUpperCase()}-${new Date().getFullYear()}`;
}

function seedHistory(): HistoryEntry[] {
  const now = Date.now();
  return [
    { id: "h1", sessionId: "LS-4KX2A-2026", createdAt: new Date(now - 3 * 3600e3).toISOString(), phases: 7, processingSeconds: 184, status: "Completed" },
    { id: "h2", sessionId: "LS-9PQ1D-2026", createdAt: new Date(now - 4 * 864e5).toISOString(), phases: 7, processingSeconds: 201, status: "Completed" },
    { id: "h3", sessionId: "LS-2ZB7M-2026", createdAt: new Date(now - 20 * 864e5).toISOString(), phases: 7, processingSeconds: 176, status: "Completed" },
  ];
}

export function formatBytes(bytes: number) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}
