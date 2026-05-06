/**
 * Lightweight skeleton shown while a lazy-loaded route chunk is downloading.
 * Uses CSS-only shimmer so no extra deps and no JS work on the main thread.
 */
export function RouteFallback({ label = 'Loading view' }: { label?: string }) {
  return (
    <div
      role="status"
      aria-label={label}
      className="mx-auto my-12 w-full max-w-3xl space-y-4 px-6"
    >
      <div className="h-8 w-1/3 animate-pulse rounded-md bg-slate-200/70 dark:bg-slate-800/70" />
      <div className="h-4 w-2/3 animate-pulse rounded bg-slate-200/70 dark:bg-slate-800/70" />
      <div className="grid grid-cols-1 gap-3 md:grid-cols-3">
        {[0, 1, 2].map((i) => (
          <div
            key={i}
            className="h-32 animate-pulse rounded-2xl bg-slate-200/70 dark:bg-slate-800/70"
          />
        ))}
      </div>
      <div className="h-64 animate-pulse rounded-2xl bg-slate-200/70 dark:bg-slate-800/70" />
      <span className="sr-only">{label}…</span>
    </div>
  );
}
