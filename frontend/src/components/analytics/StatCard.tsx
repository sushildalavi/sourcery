import type { ReactNode } from 'react';
import { motion } from 'framer-motion';
import { Card } from '../ui/Card';
import { cn } from '../../lib/cn';

interface StatCardProps {
  label: string;
  value: ReactNode;
  hint?: string;
  accent?: boolean;
  delay?: number;
}

export function StatCard({ label, value, hint, accent, delay = 0 }: StatCardProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.25, delay }}
    >
      <Card className={cn('p-5', accent && 'border-amber-500/40 bg-amber-500/5 dark:bg-amber-500/10')}>
        <div className="text-[10px] font-semibold uppercase tracking-[0.14em] text-zinc-500 dark:text-zinc-400">
          {label}
        </div>
        <div className="mt-1 font-display text-3xl italic text-zinc-900 dark:text-zinc-50">{value}</div>
        {hint && <div className="mt-1 text-[11px] text-zinc-500 dark:text-zinc-500">{hint}</div>}
      </Card>
    </motion.div>
  );
}
