import { NavLink, useLocation } from 'react-router-dom';
import { motion } from 'framer-motion';
import { Sparkles, Library, Globe, ChartBar, X } from 'lucide-react';
import type { ComponentType, SVGProps } from 'react';
import { ThemeToggle } from './ThemeToggle';
import { cn } from '../../lib/cn';

type NavItem = {
  to: string;
  label: string;
  Icon: ComponentType<SVGProps<SVGSVGElement> & { size?: number | string }>;
  hint: string;
};

const NAV_ITEMS: NavItem[] = [
  { to: '/', label: 'Uploaded', Icon: Library, hint: 'Your docs' },
  { to: '/public', label: 'Public', Icon: Globe, hint: 'Open literature' },
  { to: '/analytics', label: 'Analytics', Icon: ChartBar, hint: 'Eval metrics' },
];

interface SidebarProps {
  open?: boolean;
  onClose?: () => void;
}

export function Sidebar({ open = true, onClose }: SidebarProps) {
  const location = useLocation();

  const content = (
    <nav className="flex h-full w-full flex-col gap-1 p-3">
      {/* Brand */}
      <div className="mb-3 flex items-center gap-2.5 px-2 py-2">
        <span
          className="inline-flex h-8 w-8 items-center justify-center rounded-xl bg-amber-500 text-zinc-950 shadow-sm"
          aria-hidden
        >
          <Sparkles size={16} />
        </span>
        <div className="leading-tight">
          <div className="font-display text-[1.05rem] italic tracking-tight text-zinc-900 dark:text-zinc-50">
            ScholarRAG
          </div>
          <div className="text-[10px] uppercase tracking-wider text-zinc-500 dark:text-zinc-400">
            Research assistant
          </div>
        </div>
        {onClose && (
          <button
            type="button"
            onClick={onClose}
            className="ml-auto rounded-lg p-1 text-zinc-500 hover:bg-zinc-100 dark:hover:bg-zinc-800 lg:hidden"
            aria-label="Close sidebar"
          >
            <X size={16} />
          </button>
        )}
      </div>

      <div className="px-2 pb-2 text-[10px] font-semibold uppercase tracking-wider text-zinc-500 dark:text-zinc-500">
        Workspaces
      </div>

      <div className="flex flex-col gap-1">
        {NAV_ITEMS.map((item) => {
          const active = location.pathname === item.to;
          return (
            <NavLink
              key={item.to}
              to={item.to}
              onClick={onClose}
              className={cn(
                'group relative flex items-center gap-3 rounded-xl px-3 py-2 text-sm font-medium transition',
                active
                  ? 'text-zinc-900 dark:text-zinc-50'
                  : 'text-zinc-600 hover:text-zinc-900 hover:bg-zinc-100 dark:text-zinc-300 dark:hover:text-zinc-50 dark:hover:bg-zinc-800/70',
              )}
            >
              {active && (
                <motion.span
                  layoutId="active-mode-pill"
                  className="absolute inset-0 rounded-xl bg-amber-500/10 ring-1 ring-amber-500/40 dark:bg-amber-500/15"
                  transition={{ type: 'spring', stiffness: 420, damping: 32 }}
                  aria-hidden
                />
              )}
              <motion.span
                className={cn(
                  'relative inline-flex h-7 w-7 items-center justify-center rounded-lg',
                  active
                    ? 'bg-amber-500 text-zinc-950'
                    : 'bg-zinc-100 text-zinc-600 dark:bg-zinc-800 dark:text-zinc-300',
                )}
                animate={{ scale: active ? 1 : 0.98 }}
                transition={{ duration: 0.2 }}
              >
                <item.Icon size={14} />
              </motion.span>
              <span className="relative flex flex-col">
                <span>{item.label}</span>
                <span className="text-[10px] font-normal text-zinc-500 dark:text-zinc-500">
                  {item.hint}
                </span>
              </span>
            </NavLink>
          );
        })}
      </div>

      <div className="mt-auto flex items-center justify-between rounded-xl border border-zinc-200 bg-white/70 p-2 dark:border-zinc-800 dark:bg-zinc-900/70">
        <div className="flex flex-col pl-2">
          <span className="text-xs font-medium text-zinc-800 dark:text-zinc-100">Theme</span>
          <span className="text-[10px] text-zinc-500 dark:text-zinc-400">Dark / light</span>
        </div>
        <ThemeToggle />
      </div>
    </nav>
  );

  return (
    <>
      {/* Desktop */}
      <aside className="hidden lg:flex h-full w-64 shrink-0 border-r border-zinc-200 bg-zinc-50/80 backdrop-blur dark:border-zinc-800 dark:bg-zinc-950/80">
        {content}
      </aside>

      {/* Mobile drawer */}
      {open && (
        <div className="lg:hidden fixed inset-0 z-50 flex">
          <div
            className="absolute inset-0 bg-black/50 backdrop-blur-sm"
            onClick={onClose}
            aria-hidden
          />
          <motion.aside
            initial={{ x: -280 }}
            animate={{ x: 0 }}
            exit={{ x: -280 }}
            transition={{ type: 'spring', stiffness: 420, damping: 36 }}
            className="relative flex h-full w-72 border-r border-zinc-200 bg-zinc-50 dark:border-zinc-800 dark:bg-zinc-950"
          >
            {content}
          </motion.aside>
        </div>
      )}
    </>
  );
}
