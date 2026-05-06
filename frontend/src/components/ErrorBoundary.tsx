import { Component, type ErrorInfo, type ReactNode } from 'react';

interface Props {
  children: ReactNode;
  /** Optional render function for the fallback UI. */
  fallback?: (error: Error, reset: () => void) => ReactNode;
}

interface State {
  error: Error | null;
}

export class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null };

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    // Surface in dev console; production deployments wire this into a
    // proper error reporter (Sentry, etc.) at the entry point.
    console.error('[ErrorBoundary]', error, info.componentStack);
  }

  reset = () => this.setState({ error: null });

  render() {
    const { error } = this.state;
    if (!error) return this.props.children;

    if (this.props.fallback) {
      return this.props.fallback(error, this.reset);
    }

    return (
      <div
        role="alert"
        className="mx-auto my-12 max-w-xl rounded-2xl border border-rose-200 bg-rose-50 p-6 text-rose-900 shadow-sm dark:border-rose-900/50 dark:bg-rose-950/40 dark:text-rose-100"
      >
        <h2 className="text-lg font-semibold">Something went sideways.</h2>
        <p className="mt-2 text-sm opacity-90">
          The view crashed while rendering. The rest of the app is still
          alive — try retrying, or jump back to the home view.
        </p>
        <pre className="mt-3 overflow-x-auto rounded bg-rose-100/70 p-3 text-xs dark:bg-rose-900/40">
          {error.message}
        </pre>
        <div className="mt-4 flex gap-2">
          <button
            type="button"
            onClick={this.reset}
            className="rounded-lg bg-rose-600 px-3 py-1.5 text-sm font-medium text-white hover:bg-rose-700"
          >
            Retry
          </button>
          <a
            href="/"
            className="rounded-lg border border-rose-300 px-3 py-1.5 text-sm font-medium hover:bg-rose-100 dark:border-rose-800 dark:hover:bg-rose-900/40"
          >
            Go home
          </a>
        </div>
      </div>
    );
  }
}
