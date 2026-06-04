import { lazy, Suspense } from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import { Layout } from './components/layout/Layout';
import { ErrorBoundary } from './components/ErrorBoundary';
import { RouteFallback } from './components/RouteFallback';
import UploadedChat from './routes/UploadedChat';
import PublicChat from './routes/PublicChat';

// Analytics is the heaviest route (charts) — defer it so the chat shell
// hydrates fast on first paint.
const Analytics = lazy(() => import('./routes/Analytics'));
const AgenticResearch = lazy(() => import('./routes/AgenticResearch'));

export default function App() {
  return (
    <ErrorBoundary>
      <Routes>
        <Route element={<Layout />}>
          <Route index element={<UploadedChat />} />
          <Route path="/public" element={<PublicChat />} />
          <Route
            path="/agent"
            element={
              <Suspense fallback={<RouteFallback label="Loading agentic research" />}>
                <AgenticResearch />
              </Suspense>
            }
          />
          <Route
            path="/analytics"
            element={
              <Suspense fallback={<RouteFallback label="Loading analytics" />}>
                <Analytics />
              </Suspense>
            }
          />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Route>
      </Routes>
    </ErrorBoundary>
  );
}
