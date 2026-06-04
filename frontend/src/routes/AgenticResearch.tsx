import { useMemo, useState } from 'react';
import { motion } from 'framer-motion';
import { Loader, Sparkles, Bot } from 'lucide-react';
import { api } from '../api/client';
import type { AgentResearchResponse } from '../api/types';
import { Button } from '../components/ui/Button';
import { Badge } from '../components/ui/Badge';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../components/ui/Card';

const PROMPTS = [
  'What evidence supports retrieval-augmented generation for reducing hallucinations?',
  'Compare dense retrieval and sparse retrieval for scholarly search.',
  'Summarize the evidence for hybrid retrieval in citation-grounded QA.',
  'What are the strongest limitations of pure dense retrieval?',
];

function confidenceTone(confidence: number): 'support' | 'warn' | 'accent' {
  if (confidence >= 0.8) return 'support';
  if (confidence >= 0.6) return 'accent';
  return 'warn';
}

function fmtConfidence(value: number | undefined | null): string {
  if (value == null || Number.isNaN(value)) return '—';
  return `${Math.round(value * 100)}%`;
}

export default function AgenticResearch() {
  const [query, setQuery] = useState(PROMPTS[0]);
  const [scope, setScope] = useState<'uploaded' | 'public' | 'both'>('both');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [result, setResult] = useState<AgentResearchResponse | null>(null);

  const evidenceCount = result?.evidence?.length ?? 0;
  const topEvidence = useMemo(() => result?.evidence?.slice(0, 6) ?? [], [result]);

  const run = async (nextQuery = query) => {
    const trimmed = nextQuery.trim();
    if (!trimmed || loading) return;
    setLoading(true);
    setError('');
    try {
      const res = await api.runAgentResearch({
        query: trimmed,
        scope,
        limit: 6,
        use_llm: true,
        allow_general_background: scope === 'both',
      });
      setResult(res);
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : 'Failed to run agentic research';
      setError(msg);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="h-full overflow-y-auto">
      <div className="mx-auto flex max-w-6xl flex-col gap-6 px-5 py-6 lg:px-8">
        <motion.div
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3 }}
          className="flex flex-col gap-3 border-b border-zinc-200 pb-5 sm:flex-row sm:items-end sm:justify-between dark:border-zinc-800"
        >
          <div className="space-y-1">
            <div className="text-[10px] font-semibold uppercase tracking-[0.18em] text-amber-600 dark:text-amber-400">
              Agentic RAG
            </div>
            <h1 className="font-display text-4xl italic tracking-tight text-zinc-900 dark:text-zinc-50">
              Planner, retriever, verifier
            </h1>
            <p className="max-w-2xl text-sm text-zinc-500 dark:text-zinc-400">
              Run the agentic research workflow end to end: plan retrieval, fetch evidence, rerank, answer, and verify support.
            </p>
          </div>
          <div className="flex items-center gap-2 text-[11px] text-zinc-500 dark:text-zinc-400">
            <Bot size={14} className="text-amber-500" />
            MCP-ready tool surface
          </div>
        </motion.div>

        <div className="grid gap-4 lg:grid-cols-[1.1fr_0.9fr]">
          <Card>
            <CardHeader>
              <CardTitle>Query</CardTitle>
              <CardDescription>Ask for grounded synthesis, comparison, or evidence-first answers.</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <textarea
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                rows={6}
                className="w-full rounded-2xl border border-zinc-200 bg-white px-4 py-3 text-sm leading-relaxed text-zinc-900 outline-none transition placeholder:text-zinc-400 focus:border-amber-500/50 dark:border-zinc-800 dark:bg-zinc-950/60 dark:text-zinc-50 dark:placeholder:text-zinc-500"
                placeholder="Ask an evidence-grounded question..."
              />

              <div className="flex flex-wrap gap-2">
                {PROMPTS.map((prompt) => (
                  <button
                    key={prompt}
                    type="button"
                    onClick={() => setQuery(prompt)}
                    className="rounded-full border border-zinc-200 bg-white px-3 py-1.5 text-xs text-zinc-700 transition hover:border-amber-500/40 hover:bg-amber-500/5 dark:border-zinc-800 dark:bg-zinc-900 dark:text-zinc-300"
                  >
                    {prompt}
                  </button>
                ))}
              </div>

              <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
                <label className="flex flex-col gap-1 text-xs text-zinc-500 dark:text-zinc-400">
                  Scope
                  <select
                    value={scope}
                    onChange={(e) => setScope(e.target.value as 'uploaded' | 'public' | 'both')}
                    className="rounded-xl border border-zinc-200 bg-white px-3 py-2 text-sm text-zinc-900 outline-none dark:border-zinc-800 dark:bg-zinc-950/60 dark:text-zinc-50"
                  >
                    <option value="both">Uploaded + public</option>
                    <option value="uploaded">Uploaded only</option>
                    <option value="public">Public only</option>
                  </select>
                </label>

                <Button onClick={() => void run()} disabled={loading} className="min-w-[140px]">
                  {loading ? <Loader size={14} className="animate-spin" /> : <Sparkles size={14} />}
                  {loading ? 'Running' : 'Run agent'}
                </Button>
              </div>

              {error && (
                <div className="rounded-2xl border border-red-500/30 bg-red-500/5 px-4 py-3 text-sm text-red-700 dark:text-red-300">
                  {error}
                </div>
              )}
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Result</CardTitle>
              <CardDescription>{result ? 'Latest agentic run' : 'No run yet'}</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {result ? (
                <>
                  <div className="flex flex-wrap items-center gap-2">
                    <Badge tone={confidenceTone(result.confidence)}>Confidence {fmtConfidence(result.confidence)}</Badge>
                    {result.needs_human_review && <Badge tone="warn">Needs human review</Badge>}
                    <Badge tone="muted">{evidenceCount} evidence item{evidenceCount === 1 ? '' : 's'}</Badge>
                  </div>

                  <div className="rounded-2xl border border-zinc-200 bg-zinc-50/80 p-4 text-sm leading-relaxed text-zinc-800 dark:border-zinc-800 dark:bg-zinc-950/40 dark:text-zinc-100 whitespace-pre-wrap">
                    {result.answer}
                  </div>

                  <div className="flex flex-wrap gap-2">
                    {(result.citations || []).map((cite) => (
                      <Badge key={cite} tone="accent">
                        {cite}
                      </Badge>
                    ))}
                  </div>

                  {result.plan && (
                    <div className="grid grid-cols-2 gap-2 text-xs text-zinc-500 dark:text-zinc-400">
                      <div>Intent: {result.plan.intent}</div>
                      <div>Strategy: {result.plan.source_strategy}</div>
                      <div>Evidence target: {result.plan.required_evidence_count}</div>
                      <div>Scope: {result.plan.scope_hint}</div>
                    </div>
                  )}

                  {topEvidence.length > 0 && (
                    <div className="space-y-2">
                      <div className="text-xs font-semibold uppercase tracking-wider text-zinc-500 dark:text-zinc-400">
                        Evidence
                      </div>
                      <div className="space-y-2">
                        {topEvidence.map((item) => (
                          <div
                            key={item.source_id}
                            className="rounded-2xl border border-zinc-200 bg-white p-3 text-sm dark:border-zinc-800 dark:bg-zinc-900/60"
                          >
                            <div className="flex flex-wrap items-center gap-2">
                              <span className="font-medium text-zinc-900 dark:text-zinc-100">{item.title}</span>
                              <Badge tone="muted">{item.source}</Badge>
                              <Badge tone={item.source === 'uploaded' ? 'support' : 'accent'}>
                                {item.citation || item.source_id}
                              </Badge>
                            </div>
                            <div className="mt-2 text-xs leading-relaxed text-zinc-600 dark:text-zinc-400">
                              {item.snippet || 'No snippet available.'}
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </>
              ) : (
                <div className="rounded-2xl border border-dashed border-zinc-300 bg-zinc-50/60 p-6 text-center text-sm text-zinc-500 dark:border-zinc-800 dark:bg-zinc-950/30 dark:text-zinc-400">
                  Run a query to see the agent plan, evidence, and verification output.
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}

