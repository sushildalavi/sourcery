import type { HTMLAttributes } from 'react';
import { cn } from '../../lib/cn';

type BadgeTone = 'neutral' | 'accent' | 'support' | 'warn' | 'muted';

const toneClasses: Record<BadgeTone, string> = {
  neutral:
    'bg-zinc-100 text-zinc-700 border-zinc-200 dark:bg-zinc-800 dark:text-zinc-200 dark:border-zinc-700',
  accent:
    'bg-amber-500/15 text-amber-700 border-amber-500/30 dark:text-amber-300',
  support:
    'bg-emerald-500/15 text-emerald-700 border-emerald-500/30 dark:text-emerald-300',
  warn:
    'bg-rose-500/15 text-rose-700 border-rose-500/30 dark:text-rose-300',
  muted:
    'bg-zinc-50 text-zinc-500 border-zinc-200 dark:bg-zinc-900 dark:text-zinc-400 dark:border-zinc-800',
};

interface BadgeProps extends HTMLAttributes<HTMLSpanElement> {
  tone?: BadgeTone;
}

export function Badge({ className, tone = 'neutral', ...rest }: BadgeProps) {
  return (
    <span
      className={cn(
        'inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-[0.7rem] font-medium',
        toneClasses[tone],
        className,
      )}
      {...rest}
    />
  );
}
