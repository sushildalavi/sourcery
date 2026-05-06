import type { MsaCalibrationLatest } from '../../api/types';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '../ui/Card';
import { Badge } from '../ui/Badge';

interface CalibrationTableProps {
  calibration: MsaCalibrationLatest | null;
  modeLabel?: string;
}

function fmt(n: number | undefined | null, digits = 3): string {
  if (n == null || Number.isNaN(n)) return '—';
  return Number(n).toFixed(digits);
}

function fmtPct(n: number | undefined | null): string {
  if (n == null || Number.isNaN(n)) return '—';
  return `${Math.round(Number(n) * 100)}%`;
}

export function CalibrationTable({ calibration, modeLabel }: CalibrationTableProps) {
  const metrics = (calibration?.metrics || {}) as Record<string, number | undefined>;
  return (
    <Card>
      <CardHeader>
        <CardTitle>{modeLabel ? `${modeLabel} — calibration` : 'MSA calibration weights'}</CardTitle>
        <CardDescription>
          Logistic over M (meaning), S (stance), A (attribution). Used at inference time to score confidence.
        </CardDescription>
      </CardHeader>
      <CardContent>
        {!calibration ? (
          <div className="py-6 text-center text-sm text-zinc-500 dark:text-zinc-400">
            No calibration recorded for this mode yet.
          </div>
        ) : (
          <div className="flex flex-col gap-3">
            <div className="flex flex-wrap items-center gap-1.5">
              <Badge tone="accent">{calibration.model_name}</Badge>
              <Badge tone="muted">{calibration.dataset_size} claims</Badge>
            </div>
            <div className="grid grid-cols-4 gap-2">
              {[
                ['w1 (M)', calibration.weights?.w1],
                ['w2 (S)', calibration.weights?.w2],
                ['w3 (A)', calibration.weights?.w3],
                ['bias', calibration.weights?.b],
              ].map(([label, value]) => (
                <div
                  key={label as string}
                  className="rounded-xl border border-zinc-200 bg-zinc-50/60 p-2.5 text-center dark:border-zinc-800 dark:bg-zinc-900/60"
                >
                  <div className="text-[10px] font-semibold uppercase tracking-wider text-zinc-500">
                    {label}
                  </div>
                  <div className="mt-0.5 font-mono text-sm text-zinc-900 dark:text-zinc-50">
                    {fmt(value as number | undefined)}
                  </div>
                </div>
              ))}
            </div>
            {(metrics.auc != null || metrics.brier != null || metrics.accuracy != null) && (
              <div className="grid grid-cols-3 gap-2">
                {metrics.auc != null && (
                  <div className="rounded-xl border border-zinc-200 bg-zinc-50/60 p-2 text-center dark:border-zinc-800 dark:bg-zinc-900/60">
                    <div className="text-[10px] font-semibold uppercase tracking-wider text-zinc-500">AUC</div>
                    <div className="mt-0.5 font-mono text-xs text-zinc-900 dark:text-zinc-50">{fmt(metrics.auc, 3)}</div>
                  </div>
                )}
                {metrics.brier != null && (
                  <div className="rounded-xl border border-zinc-200 bg-zinc-50/60 p-2 text-center dark:border-zinc-800 dark:bg-zinc-900/60">
                    <div className="text-[10px] font-semibold uppercase tracking-wider text-zinc-500">Brier</div>
                    <div className="mt-0.5 font-mono text-xs text-zinc-900 dark:text-zinc-50">{fmt(metrics.brier, 3)}</div>
                  </div>
                )}
                {metrics.accuracy != null && (
                  <div className="rounded-xl border border-zinc-200 bg-zinc-50/60 p-2 text-center dark:border-zinc-800 dark:bg-zinc-900/60">
                    <div className="text-[10px] font-semibold uppercase tracking-wider text-zinc-500">Acc</div>
                    <div className="mt-0.5 font-mono text-xs text-zinc-900 dark:text-zinc-50">{fmtPct(metrics.accuracy)}</div>
                  </div>
                )}
              </div>
            )}
            {calibration.created_at && (
              <div className="text-[11px] text-zinc-500 dark:text-zinc-500">
                Fit {new Date(calibration.created_at).toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' })}
              </div>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
