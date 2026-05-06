import { Menu, Sparkles } from 'lucide-react';
import { ThemeToggle } from './ThemeToggle';

interface MobileTopBarProps {
  onOpenSidebar: () => void;
}

export function MobileTopBar({ onOpenSidebar }: MobileTopBarProps) {
  return (
    <header className="lg:hidden sticky top-0 z-30 flex h-14 items-center justify-between border-b border-zinc-200 bg-white/85 px-3 backdrop-blur dark:border-zinc-800 dark:bg-zinc-950/85">
      <button
        type="button"
        onClick={onOpenSidebar}
        aria-label="Open sidebar"
        className="inline-flex h-9 w-9 items-center justify-center rounded-xl text-zinc-700 hover:bg-zinc-100 dark:text-zinc-200 dark:hover:bg-zinc-800"
      >
        <Menu size={18} />
      </button>
      <div className="flex items-center gap-2">
        <span
          className="inline-flex h-7 w-7 items-center justify-center rounded-lg bg-amber-500 text-zinc-950"
          aria-hidden
        >
          <Sparkles size={14} />
        </span>
        <span className="font-display italic text-zinc-900 dark:text-zinc-50">ScholarRAG</span>
      </div>
      <ThemeToggle />
    </header>
  );
}
