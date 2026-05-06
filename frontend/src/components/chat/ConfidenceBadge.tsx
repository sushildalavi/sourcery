import type { ConfidenceObject } from '../../api/types';
import { Badge } from '../ui/Badge';

interface ConfidenceBadgeProps {
  confidence?: ConfidenceObject;
  showWhenMissing?: boolean;
}

export function ConfidenceBadge({ confidence, showWhenMissing = false }: ConfidenceBadgeProps) {
  if (!confidence) {
    return showWhenMissing ? <Badge tone="muted">Confidence: N/A</Badge> : null;
  }
  if (confidence.needs_clarification) {
    return <Badge tone="warn">Confidence: Clarify</Badge>;
  }
  const pct = Math.round((confidence.score || 0) * 100);
  const raw = (confidence.label || 'Low').toLowerCase();
  const tone = raw === 'high' ? 'accent' : raw === 'med' ? 'neutral' : 'warn';
  const f = confidence.factors;
  const msa = f?.msa;
  const title = [
    'Confidence score = calibrated P(claim correctly supported | M, S, A).',
    'Logistic regression: sigmoid(b + w1·M + w2·S + w3·A).',
    'M = entailment probability (NLI) · S = retrieval stability · A = multi-source agreement.',
    msa
      ? `This answer: M=${msa.M.toFixed(2)} S=${msa.S.toFixed(2)} A=${msa.A.toFixed(2)}`
      : (f
          ? `Retrieval-only fallback: sim=${f.top_sim.toFixed(2)} cov=${f.citation_coverage.toFixed(2)} margin=${f.evidence_margin.toFixed(2)}`
          : ''),
  ].filter(Boolean).join('\n');
  return (
    <Badge tone={tone} title={title}>
      Confidence: {confidence.label} {pct}%
    </Badge>
  );
}
