import { motion } from 'framer-motion';
import { ArrowUpRight, Sparkles } from 'lucide-react';
import { cn } from '../../lib/cn';

interface FollowUpChipsProps {
  suggestions: string[];
  onSelect: (prompt: string) => void;
  onNewThread?: () => void;
  disabled?: boolean;
}

/**
 * Suggestion chips rendered beneath a settled assistant reply. Click sends
 * as a new user turn; the `NEW THREAD` button starts a fresh conversation.
 */
export function FollowUpChips({
  suggestions,
  onSelect,
  onNewThread,
  disabled = false,
}: FollowUpChipsProps) {
  if (!suggestions.length && !onNewThread) return null;

  return (
    <motion.div
      initial={{ opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.28, delay: 0.1 }}
      className="mt-3 flex flex-wrap items-center gap-1.5"
    >
      {suggestions.slice(0, 3).map((s, i) => (
        <button
          key={i}
          type="button"
          disabled={disabled}
          onClick={() => onSelect(s)}
          className={cn(
            'group inline-flex items-center gap-1.5 rounded-full border border-zinc-200/80 bg-white/70 px-3 py-1.5 text-[11.5px] font-medium text-zinc-700 transition hover:border-amber-500/40 hover:text-zinc-900 hover:shadow-sm dark:border-zinc-800/80 dark:bg-zinc-900/50 dark:text-zinc-200 dark:hover:border-amber-500/30 dark:hover:text-zinc-50',
            disabled && 'opacity-50 cursor-not-allowed',
          )}
        >
          <Sparkles size={11} className="text-amber-500/70 transition group-hover:text-amber-500" />
          <span className="truncate">{s}</span>
        </button>
      ))}
      {onNewThread && (
        <button
          type="button"
          onClick={onNewThread}
          disabled={disabled}
          className="ml-auto inline-flex items-center gap-1 rounded-full px-2 py-1 text-[10px] font-semibold uppercase tracking-[0.14em] text-zinc-400 transition hover:text-amber-600 dark:text-zinc-600 dark:hover:text-amber-400"
        >
          new thread
          <ArrowUpRight size={11} />
        </button>
      )}
    </motion.div>
  );
}
