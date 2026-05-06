import { AnimatePresence, motion } from 'framer-motion';
import { Moon, Sun } from 'lucide-react';
import { useTheme } from '../../hooks/useTheme';
import { cn } from '../../lib/cn';

interface ThemeToggleProps {
  className?: string;
}

export function ThemeToggle({ className }: ThemeToggleProps) {
  const { theme, toggle } = useTheme();
  const isDark = theme === 'dark';

  return (
    <button
      type="button"
      onClick={toggle}
      aria-label={isDark ? 'Switch to light theme' : 'Switch to dark theme'}
      className={cn(
        'relative inline-flex h-9 w-9 items-center justify-center rounded-xl border transition',
        'border-zinc-200 bg-white hover:bg-zinc-100 text-zinc-700',
        'dark:border-zinc-800 dark:bg-zinc-900 dark:hover:bg-zinc-800 dark:text-zinc-200',
        'focus:outline-none focus-visible:ring-2 focus-visible:ring-amber-500/60',
        className,
      )}
    >
      <AnimatePresence mode="wait" initial={false}>
        {isDark ? (
          <motion.span
            key="moon"
            initial={{ opacity: 0, rotate: -90, scale: 0.6 }}
            animate={{ opacity: 1, rotate: 0, scale: 1 }}
            exit={{ opacity: 0, rotate: 90, scale: 0.6 }}
            transition={{ duration: 0.2 }}
            className="flex"
          >
            <Moon size={16} />
          </motion.span>
        ) : (
          <motion.span
            key="sun"
            initial={{ opacity: 0, rotate: 90, scale: 0.6 }}
            animate={{ opacity: 1, rotate: 0, scale: 1 }}
            exit={{ opacity: 0, rotate: -90, scale: 0.6 }}
            transition={{ duration: 0.2 }}
            className="flex"
          >
            <Sun size={16} />
          </motion.span>
        )}
      </AnimatePresence>
    </button>
  );
}
