import { motion } from 'framer-motion';
import { ArrowRight, BookOpen, Clock, Compass, Sparkles, TrendingUp } from 'lucide-react';
import { cn } from '../../lib/cn';

interface QuickAction {
  eyebrow: string;
  title: string;
  description: string;
  prompt: string;
}

interface HomeHeroProps {
  mode: 'uploaded' | 'public';
  title: string;
  description: string;
  prompts: QuickAction[];
  onSelectPrompt: (prompt: string) => void;
  paperCount?: number;
  providersOnline?: string[];
}

const ICONS = [Clock, TrendingUp, Compass, BookOpen];

export function HomeHero({
  mode,
  title: _title,
  description: _description,
  prompts,
  onSelectPrompt,
  paperCount,
  providersOnline = ['S2', 'OpenAlex', 'arXiv', 'Crossref'],
}: HomeHeroProps) {
  const displayPrompts = prompts.slice(0, 4);

  return (
    <div className="relative mx-auto flex w-full max-w-3xl flex-col items-center justify-center gap-10 px-2 py-12 text-center">
      {/* providers pill */}
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4 }}
        className="flex items-center gap-2 rounded-full border border-zinc-200/70 bg-white/60 px-3 py-1 text-[10px] font-semibold uppercase tracking-[0.15em] text-zinc-600 backdrop-blur dark:border-zinc-800/80 dark:bg-zinc-900/60 dark:text-zinc-300"
      >
        <span className="relative flex h-2 w-2 items-center justify-center">
          <span className="absolute h-2 w-2 animate-ping rounded-full bg-amber-400/70" />
          <span className="relative h-1.5 w-1.5 rounded-full bg-amber-500" />
        </span>
        {paperCount != null && <span>{paperCount} papers</span>}
        {paperCount != null && <span className="text-zinc-400 dark:text-zinc-600">·</span>}
        <span>{providersOnline.length} providers online</span>
      </motion.div>

      {/* hero headline */}
      <motion.div
        initial={{ opacity: 0, y: 14 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.55, ease: 'easeOut', delay: 0.05 }}
        className="flex flex-col items-center gap-2"
      >
        <h1 className="font-display text-5xl leading-[1.05] tracking-tight text-zinc-900 sm:text-6xl dark:text-zinc-50">
          Ask the literature,
        </h1>
        <h1 className="font-display text-5xl italic leading-[1.05] tracking-tight sm:text-6xl">
          <span className="bg-gradient-to-r from-amber-500 via-orange-400 to-amber-300 bg-clip-text text-transparent">
            cite with confidence.
          </span>
        </h1>
        <p className="mt-4 max-w-xl text-sm leading-relaxed text-zinc-600 dark:text-zinc-400">
          Grounded answers with inline citations, retrieval-quality metrics, and
          excerpt-level traceability.
        </p>
      </motion.div>

      {/* quick action cards */}
      <div className="grid w-full gap-3 sm:grid-cols-2">
        {displayPrompts.map((p, i) => {
          const Icon = ICONS[i % ICONS.length];
          return (
            <motion.button
              key={p.title}
              type="button"
              onClick={() => p.prompt && onSelectPrompt(p.prompt)}
              disabled={!p.prompt}
              initial={{ opacity: 0, y: 14 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.35, delay: 0.15 + i * 0.07 }}
              whileHover={p.prompt ? { y: -3 } : undefined}
              whileTap={p.prompt ? { scale: 0.98 } : undefined}
              className={cn(
                'group relative overflow-hidden rounded-2xl border border-zinc-200/80 bg-white/70 p-4 text-left transition hover:border-amber-500/50 hover:shadow-lg hover:shadow-amber-500/5 backdrop-blur',
                'dark:border-zinc-800/80 dark:bg-zinc-900/60 dark:hover:border-amber-500/40',
                !p.prompt && 'opacity-50 cursor-not-allowed',
              )}
            >
              <div className="flex items-start gap-3">
                <span className="inline-flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-amber-500/10 text-amber-600 ring-1 ring-amber-500/20 transition group-hover:bg-amber-500/15 dark:bg-amber-500/15 dark:text-amber-300 dark:ring-amber-500/30">
                  <Icon size={16} />
                </span>
                <div className="flex-1">
                  <div className="flex items-center justify-between gap-2">
                    <div className="text-sm font-semibold text-zinc-900 dark:text-zinc-50">
                      {p.title}
                    </div>
                    <ArrowRight
                      size={13}
                      className="text-zinc-400 transition group-hover:translate-x-0.5 group-hover:text-amber-500 dark:text-zinc-600 dark:group-hover:text-amber-400"
                    />
                  </div>
                  <div className="mt-1 text-[11.5px] leading-relaxed text-zinc-500 dark:text-zinc-400">
                    {p.description}
                  </div>
                </div>
              </div>
            </motion.button>
          );
        })}
      </div>

      {/* palette / mode hint */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.5, delay: 0.4 }}
        className="flex items-center gap-1.5 text-[10px] font-medium uppercase tracking-[0.14em] text-zinc-400 dark:text-zinc-600"
      >
        <Sparkles size={11} className="text-amber-500/70" />
        {mode === 'public' ? 'public research mode' : 'uploaded documents mode'}
        <span className="text-zinc-300 dark:text-zinc-700">·</span>
        <span>{providersOnline.slice(0, 4).join(' · ')}</span>
      </motion.div>
    </div>
  );
}
