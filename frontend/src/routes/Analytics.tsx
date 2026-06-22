import { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import { Loader, RefreshCw } from 'lucide-react';
import { api, API_BASE } from '../api/client';
import type {
  EvalRunResponse,
  JudgeRunSummary,
  MsaCalibrationLatest,
} from '../api/types';
import { StatCard } from '../components/analytics/StatCard';
import { RecallChart } from '../components/analytics/RecallChart';
import { LatencyChart } from '../components/analytics/LatencyChart';
import { FaithfulnessChart } from '../components/analytics/FaithfulnessChart';
import { CalibrationTable } from '../components/analytics/CalibrationTable';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '../components/ui/Card';
import { Button } from '../components/ui/Button';
import { Badge } from '../components/ui/Badge';

interface AnalyticsState {
  runs: EvalRunResponse[];
  judgeRuns: JudgeRunSummary[];
  calibrationUnified: MsaCalibrationLatest | null;
  loading: boolean;
  error: string;
}

function fmt(n: number | undefined | null, digits = 3): string {
  if (n == null || Number.isNaN(n)) return '—';
  return Number(n).toFixed(digits);
}

function fmtDate(raw?: string | null): string {
  if (!raw) return '—';
  try {
    const d = new Date(raw);
    return `${d.toLocaleDateString(undefined, { month: 'short', day: 'numeric' })} ${d.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' })}`;
  } catch {
    return raw.slice(0, 16);
  }
}

export default function Analytics() {
  const [state, setState] = useState<AnalyticsState>({
    runs: [],
    judgeRuns: [],
    calibrationUnified: null,
    loading: true,
    error: '',
  });

  const load = async () => {
    setState((s) => ({ ...s, loading: true, error: '' }));
    try {
      const [evalRes, judgeRes, calibRes] = await Promise.allSettled([
        api.listEvalRuns(20),
        api.listJudgeRuns(20),
        // Single unified calibration (pooled fit across uploaded + public modes,
        // validated by per-mode ablation: Δ Brier = 0.005).
        api.getLatestCalibration('unified'),
      ]);
      setState({
        loading: false,
        runs: evalRes.status === 'fulfilled' ? evalRes.value.runs || [] : [],
        judgeRuns: judgeRes.status === 'fulfilled' ? judgeRes.value.runs || [] : [],
        calibrationUnified: calibRes.status === 'fulfilled' ? calibRes.value || null : null,
        error:
          evalRes.status === 'rejected'
            ? evalRes.reason?.message || `Backend unreachable at ${API_BASE}`
            : '',
      });
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : 'Failed to load analytics';
      setState({
        runs: [],
        judgeRuns: [],
        calibrationUnified: null,
        loading: false,
        error: msg,
      });
    }
  };

  useEffect(() => {
    load();
  }, []);

  const latest = state.runs[0];
  const latestJudge = state.judgeRuns[0];
  const bestR5 = state.runs.reduce<number | null>((best, r) => {
    const v = r.metrics_retrieval_rerank?.recall_at?.['5'];
    if (typeof v === 'number' && (best === null || v > best)) return v;
    return best;
  }, null);
  const latestR1 = latest?.metrics_retrieval_rerank?.recall_at?.['1'];
  const latestMrr = latest?.metrics_retrieval_rerank?.mrr;
  const latestFaith = latestJudge?.metrics?.mean_overall_score;
  const latestLatency = latest
    ? Math.round(
        (latest.latency_breakdown?.retrieve_ms_avg || 0) +
          (latest.latency_breakdown?.rerank_ms_avg || 0),
      )
    : null;

  return (
    <div className="h-full overflow-y-auto">
      <div className="mx-auto flex max-w-6xl flex-col gap-5 px-5 py-6 lg:px-8">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3 }}
          className="flex flex-col justify-between gap-3 border-b border-zinc-200 pb-5 sm:flex-row sm:items-end dark:border-zinc-800"
        >
          <div className="space-y-1">
            <div className="text-[10px] font-semibold uppercase tracking-[0.18em] text-amber-600 dark:text-amber-400">
              Analytics
            </div>
            <h1 className="font-display text-4xl italic tracking-tight text-zinc-900 dark:text-zinc-50">
              evaluation metrics
            </h1>
            <p className="max-w-2xl text-sm text-zinc-500 dark:text-zinc-400">
              Read-only view of the latest retrieval, judge, and calibration results recorded in the backend.
            </p>
          </div>
          <Button variant="outline" size="sm" onClick={load} disabled={state.loading}>
            {state.loading ? <Loader size={13} className="animate-spin" /> : <RefreshCw size={13} />}
            Refresh
          </Button>
        </motion.div>

        {state.error && (
          <div className="rounded-2xl border border-red-500/30 bg-red-500/5 px-4 py-3 text-xs text-red-600 dark:text-red-300">
            {state.error}
          </div>
        )}

        {/* Stat row */}
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
          <StatCard
            label="Recall@1"
            value={fmt(latestR1)}
            hint={latest ? latest.name : 'No runs yet'}
            accent
            delay={0.05}
          />
          <StatCard
            label="MRR"
            value={fmt(latestMrr)}
            hint={bestR5 != null ? `Best R@5 ${fmt(bestR5)}` : 'Retrieval + rerank'}
            delay={0.1}
          />
          <StatCard
            label="Faithfulness"
            value={latestFaith != null ? `${Math.round(latestFaith * 100)}%` : '—'}
            hint={latestJudge ? `Run #${latestJudge.id}` : 'No judge runs'}
            delay={0.15}
          />
          <StatCard
            label="Latency"
            value={latestLatency != null ? `${latestLatency}ms` : '—'}
            hint={latest ? 'Retrieve + rerank avg' : '—'}
            delay={0.2}
          />
        </div>

        {/* Charts row */}
        <div className="grid gap-4 lg:grid-cols-2">
          {latest ? (
            <RecallChart
              recallOnly={latest.metrics_retrieval_only?.recall_at || {}}
              recallRerank={latest.metrics_retrieval_rerank?.recall_at || {}}
            />
          ) : (
            <Card>
              <CardHeader>
                <CardTitle>Recall @ K</CardTitle>
                <CardDescription>No retrieval runs available.</CardDescription>
              </CardHeader>
              <CardContent className="h-64" />
            </Card>
          )}
          {latest ? (
            <LatencyChart breakdown={latest.latency_breakdown} />
          ) : (
            <Card>
              <CardHeader>
                <CardTitle>Latency breakdown</CardTitle>
                <CardDescription>No retrieval runs available.</CardDescription>
              </CardHeader>
              <CardContent className="h-64" />
            </Card>
          )}
        </div>

        <FaithfulnessChart runs={state.judgeRuns} />

        <CalibrationTable calibration={state.calibrationUnified} modeLabel="Unified (uploaded + public)" />

        {/* Run history */}
        <Card>
          <CardHeader>
            <CardTitle>Recent retrieval runs</CardTitle>
            <CardDescription>Latest {Math.min(state.runs.length, 10)} runs stored on the backend.</CardDescription>
          </CardHeader>
          <CardContent className="overflow-x-auto">
            {state.runs.length === 0 ? (
              <div className="py-6 text-center text-sm text-zinc-500 dark:text-zinc-400">
                No retrieval runs yet.
              </div>
            ) : (
              <table className="w-full min-w-[420px] text-left text-sm">
                <thead>
                  <tr className="border-b border-zinc-200 text-xs uppercase tracking-wider text-zinc-500 dark:border-zinc-800">
                    <th className="py-2 font-medium">Run</th>
                    <th className="py-2 font-medium">When</th>
                    <th className="py-2 font-medium">Cases</th>
                    <th className="py-2 text-right font-medium">R@5</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-zinc-100 dark:divide-zinc-800">
                  {state.runs.slice(0, 10).map((r) => (
                    <tr key={`${r.run_id}-${r.created_at}`}>
                      <td className="py-2 pr-2">
                        <span className="truncate font-medium text-zinc-900 dark:text-zinc-50">{r.name}</span>
                      </td>
                      <td className="py-2 pr-2 text-xs text-zinc-500 dark:text-zinc-400">{fmtDate(r.created_at)}</td>
                      <td className="py-2 pr-2 text-xs text-zinc-500 dark:text-zinc-400">{r.case_count}</td>
                      <td className="py-2 text-right font-mono text-xs">
                        {fmt(r.metrics_retrieval_rerank?.recall_at?.['5'])}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Recent judge runs</CardTitle>
            <CardDescription>Latest judge evaluations for faithfulness coverage.</CardDescription>
          </CardHeader>
          <CardContent className="overflow-x-auto">
            {state.judgeRuns.length === 0 ? (
              <div className="py-6 text-center text-sm text-zinc-500 dark:text-zinc-400">
                No judge runs yet.
              </div>
            ) : (
              <table className="w-full min-w-[420px] text-left text-sm">
                <thead>
                  <tr className="border-b border-zinc-200 text-xs uppercase tracking-wider text-zinc-500 dark:border-zinc-800">
                    <th className="py-2 font-medium">Scope</th>
                    <th className="py-2 font-medium">Run</th>
                    <th className="py-2 font-medium">Queries</th>
                    <th className="py-2 text-right font-medium">Mean</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-zinc-100 dark:divide-zinc-800">
                  {state.judgeRuns.slice(0, 10).map((r) => (
                    <tr key={r.id}>
                      <td className="py-2 pr-2">
                        <Badge tone="neutral">{r.scope}</Badge>
                      </td>
                      <td className="py-2 pr-2 text-xs text-zinc-500 dark:text-zinc-400">#{r.id}</td>
                      <td className="py-2 pr-2 text-xs text-zinc-500 dark:text-zinc-400">{r.query_count || 0}</td>
                      <td className="py-2 text-right font-mono text-xs">
                        {Math.round((r.metrics?.mean_overall_score || 0) * 100)}%
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
