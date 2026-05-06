import { useEffect, useMemo, useRef, useState } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import {
  BarChart3,
  FileText,
  Globe,
  MessageSquarePlus,
  Search,
  Sparkles,
  type LucideIcon,
} from 'lucide-react';
import { cn } from '../../lib/cn';

export interface PaletteAction {
  id: string;
  label: string;
  description?: string;
  icon: LucideIcon;
  keywords?: string[];
  run: () => void;
}

interface CommandPaletteProps {
  actions: PaletteAction[];
}

/**
 * ⌘K / Ctrl-K quick action palette. Opens on the platform-standard shortcut,
 * filters actions on fuzzy keyword match, Enter runs the first hit.
 */
export function CommandPalette({ actions }: CommandPaletteProps) {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState('');
  const [highlight, setHighlight] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') {
        e.preventDefault();
        setOpen((v) => !v);
      }
      if (e.key === 'Escape') setOpen(false);
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);

  useEffect(() => {
    if (open) {
      setQuery('');
      setHighlight(0);
      setTimeout(() => inputRef.current?.focus(), 10);
    }
  }, [open]);

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return actions;
    return actions.filter((a) => {
      const hay = [a.label, a.description ?? '', ...(a.keywords ?? [])].join(' ').toLowerCase();
      return hay.includes(q);
    });
  }, [actions, query]);

  useEffect(() => {
    setHighlight(0);
  }, [query]);

  const onKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      setHighlight((h) => Math.min(filtered.length - 1, h + 1));
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      setHighlight((h) => Math.max(0, h - 1));
    } else if (e.key === 'Enter') {
      const choice = filtered[highlight];
      if (choice) {
        choice.run();
        setOpen(false);
      }
    }
  };

  return (
    <AnimatePresence>
      {open && (
        <motion.div
          key="palette-backdrop"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.15 }}
          className="fixed inset-0 z-[60] flex items-start justify-center bg-zinc-950/50 px-4 pt-[15vh] backdrop-blur-sm"
          onClick={() => setOpen(false)}
        >
          <motion.div
            key="palette"
            initial={{ opacity: 0, y: -8, scale: 0.98 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -8, scale: 0.98 }}
            transition={{ duration: 0.18, ease: 'easeOut' }}
            onClick={(e) => e.stopPropagation()}
            className="w-full max-w-lg overflow-hidden rounded-2xl border border-zinc-200/80 bg-white/95 shadow-2xl shadow-amber-500/10 backdrop-blur-xl dark:border-zinc-800/80 dark:bg-zinc-900/95"
          >
            <div className="flex items-center gap-2 border-b border-zinc-200/70 px-4 py-3 dark:border-zinc-800/70">
              <Search size={15} className="text-zinc-400" />
              <input
                ref={inputRef}
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyDown={onKeyDown}
                placeholder="Type a command…"
                className="flex-1 bg-transparent text-sm text-zinc-900 placeholder:text-zinc-400 focus:outline-none dark:text-zinc-50"
              />
              <span className="rounded-md border border-zinc-200 bg-zinc-100 px-1.5 py-0.5 font-mono text-[10px] text-zinc-500 dark:border-zinc-800 dark:bg-zinc-800 dark:text-zinc-400">
                ESC
              </span>
            </div>
            <div className="max-h-[50vh] overflow-y-auto p-1.5">
              {filtered.length === 0 && (
                <div className="px-4 py-6 text-center text-sm text-zinc-400">
                  No matching actions.
                </div>
              )}
              {filtered.map((a, i) => {
                const Icon = a.icon;
                return (
                  <button
                    key={a.id}
                    type="button"
                    onClick={() => {
                      a.run();
                      setOpen(false);
                    }}
                    onMouseEnter={() => setHighlight(i)}
                    className={cn(
                      'flex w-full items-center gap-3 rounded-lg px-3 py-2 text-left transition',
                      i === highlight
                        ? 'bg-amber-500/10 text-amber-700 dark:text-amber-300'
                        : 'text-zinc-700 hover:bg-zinc-100 dark:text-zinc-200 dark:hover:bg-zinc-800/80',
                    )}
                  >
                    <span className="inline-flex h-7 w-7 shrink-0 items-center justify-center rounded-lg bg-zinc-100 text-zinc-500 dark:bg-zinc-800 dark:text-zinc-300">
                      <Icon size={13} />
                    </span>
                    <span className="flex-1 min-w-0">
                      <span className="block truncate text-sm font-medium">{a.label}</span>
                      {a.description && (
                        <span className="block truncate text-[11px] text-zinc-500 dark:text-zinc-400">
                          {a.description}
                        </span>
                      )}
                    </span>
                  </button>
                );
              })}
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}

/**
 * Default palette for the main app — navigate between modes, start a new
 * conversation, open analytics. Caller can extend `extra` to pass
 * route-specific actions (e.g. clear chat, toggle evidence panel).
 */
export function useDefaultPaletteActions(extra: PaletteAction[] = []): PaletteAction[] {
  const navigate = useNavigate();
  return useMemo<PaletteAction[]>(
    () => [
      {
        id: 'nav-public',
        label: 'Switch to Public research mode',
        description: 'Query across arXiv, Semantic Scholar, OpenAlex, Crossref',
        icon: Globe,
        keywords: ['public', 'mode', 'research', 'arxiv', 's2'],
        run: () => navigate('/public'),
      },
      {
        id: 'nav-uploaded',
        label: 'Switch to My documents mode',
        description: 'Query your uploaded corpus',
        icon: FileText,
        keywords: ['uploaded', 'mode', 'docs', 'my'],
        run: () => navigate('/'),
      },
      {
        id: 'nav-analytics',
        label: 'Open Analytics',
        description: 'Calibration, latency, faithfulness dashboards',
        icon: BarChart3,
        keywords: ['analytics', 'metrics', 'calibration'],
        run: () => navigate('/analytics'),
      },
      ...extra,
    ],
    [navigate, extra],
  );
}

// Expose the common icons so callers can build their own palette entries.
export { MessageSquarePlus, Sparkles };
