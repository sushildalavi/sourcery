import { useState } from 'react';
import { Outlet, useLocation } from 'react-router-dom';
import { AnimatePresence, motion } from 'framer-motion';
import { Sidebar } from './Sidebar';
import { MobileTopBar } from './MobileTopBar';

export function Layout() {
  const [mobileOpen, setMobileOpen] = useState(false);
  const location = useLocation();

  return (
    <div className="flex h-screen w-full overflow-hidden bg-zinc-50 text-zinc-900 dark:bg-zinc-950 dark:text-zinc-50">
      <Sidebar open={mobileOpen} onClose={() => setMobileOpen(false)} />
      <div className="flex flex-1 flex-col min-w-0">
        <MobileTopBar onOpenSidebar={() => setMobileOpen(true)} />
        <main className="flex-1 min-h-0 overflow-hidden">
          <AnimatePresence mode="wait">
            <motion.div
              key={location.pathname}
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -8 }}
              transition={{ duration: 0.25, ease: 'easeOut' }}
              className="h-full min-h-0"
            >
              <Outlet />
            </motion.div>
          </AnimatePresence>
        </main>
      </div>
    </div>
  );
}
