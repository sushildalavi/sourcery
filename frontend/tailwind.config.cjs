/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './index.html',
    './src/**/*.{js,ts,jsx,tsx}',
  ],
  darkMode: 'class',
  theme: {
    extend: {
      fontFamily: {
        sans: ['Inter', 'ui-sans-serif', 'system-ui', '-apple-system', 'Segoe UI', 'Roboto', 'sans-serif'],
        display: ['"Instrument Serif"', 'Georgia', 'serif'],
        mono: ['"JetBrains Mono"', 'ui-monospace', 'SFMono-Regular', 'Menlo', 'monospace'],
      },
      colors: {
        brand: {
          bg: {
            DEFAULT: '#09090b', // zinc-950
            light: '#fafafa',   // zinc-50
          },
          surface: {
            DEFAULT: '#18181b', // zinc-900
            light: '#ffffff',
          },
          border: {
            DEFAULT: '#27272a', // zinc-800
            light: '#e4e4e7',   // zinc-200
          },
          fg: {
            DEFAULT: '#fafafa', // zinc-50
            muted: '#a1a1aa',   // zinc-400
            light: '#18181b',
            lightMuted: '#71717a', // zinc-500
          },
          accent: '#f59e0b',    // amber-500
          support: '#10b981',   // emerald-500
          warn: '#ef4444',      // red-500
        },
      },
      borderRadius: {
        '2xl': '1rem',
        '3xl': '1.25rem',
      },
      keyframes: {
        'fade-up': {
          '0%': { opacity: '0', transform: 'translateY(8px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        'pulse-dot': {
          '0%, 100%': { opacity: '0.4', transform: 'translateY(0)' },
          '50%': { opacity: '1', transform: 'translateY(-4px)' },
        },
      },
      animation: {
        'fade-up': 'fade-up 0.25s ease-out both',
        'pulse-dot': 'pulse-dot 1.2s ease-in-out infinite',
      },
    },
  },
  plugins: [],
};
