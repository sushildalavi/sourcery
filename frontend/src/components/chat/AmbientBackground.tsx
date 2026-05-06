import { motion } from 'framer-motion';

/**
 * Subtle ambient backdrop for chat / hero screens: a dark gradient vignette
 * with faint amber glow orbs and a barely-visible diagonal line field.
 * Non-interactive, pointer-events disabled, sits behind all content.
 */
export function AmbientBackground() {
  return (
    <div className="pointer-events-none absolute inset-0 -z-10 overflow-hidden">
      {/* top-left amber glow */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 0.35 }}
        transition={{ duration: 1.4, ease: 'easeOut' }}
        className="absolute -left-32 -top-32 h-[420px] w-[420px] rounded-full bg-amber-500/30 blur-[120px] dark:bg-amber-500/20"
      />
      {/* bottom-right warm glow */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 0.25 }}
        transition={{ duration: 1.4, ease: 'easeOut', delay: 0.15 }}
        className="absolute -bottom-40 -right-24 h-[520px] w-[520px] rounded-full bg-orange-600/25 blur-[140px] dark:bg-orange-600/15"
      />
      {/* dotted grid overlay */}
      <div
        className="absolute inset-0 opacity-[0.06] dark:opacity-[0.08]"
        style={{
          backgroundImage:
            'radial-gradient(circle at 1px 1px, currentColor 1px, transparent 0)',
          backgroundSize: '24px 24px',
          color: '#fff',
        }}
      />
      {/* diagonal accent lines */}
      <svg className="absolute inset-0 h-full w-full opacity-[0.04] dark:opacity-[0.08]">
        <defs>
          <pattern id="diag" width="80" height="80" patternUnits="userSpaceOnUse">
            <path d="M0 80 L80 0" stroke="currentColor" strokeWidth="0.5" fill="none" />
          </pattern>
        </defs>
        <rect width="100%" height="100%" fill="url(#diag)" className="text-amber-500" />
      </svg>
    </div>
  );
}
