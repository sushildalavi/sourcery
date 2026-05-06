import { useEffect, useMemo, useRef, useState } from 'react';
import { Link } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { ChevronDown, ExternalLink, FileText, FolderOpen, Info, Search, X } from 'lucide-react';
import type {
  Citation,
  ConfidenceObject,
  FaithfulnessReport,
  WhyTraceChunk,
} from '../../api/types';
import {
  isSkippedEntry,
  sourceDedupKey,
  type ProviderStatusEntry,
  type SkippedProviderEntry,
  type UiMessage,
} from '../../hooks/useChatSession';
import { Badge } from '../ui/Badge';
import { cn } from '../../lib/cn';

type SupportingClaim = {
  sentenceId: number;
  sentence: string;
};

type ChunkDetail = {
  citationId: number;
  page?: number;
  snippet?: string;
  evidenceId?: string;
  rerankScore?: number;
  simScore?: number;
  msaScore?: number;
  msaM?: number;
  msaS?: number;
  msaA?: number;
  cited: boolean;
  metadataOnly?: boolean;
  msaSupported?: boolean;
  claims: SupportingClaim[];
};

type SourceGroup = {
  key: string;
  displayId: number;
  title: string;
  doc_id?: number;
  url?: string;
  source?: string;
  cited: boolean;
  confidence_obj?: ConfidenceObject;
  pages: number[];
  chunks: ChunkDetail[];
  msaTop?: number;
  metadataOnly?: boolean;
  msaSupported?: boolean;
};

interface SourcesPanelProps {
  mode?: 'uploaded' | 'public';
  citations: Citation[];
  trace: WhyTraceChunk[];
  faithfulness?: FaithfulnessReport | null;
  retrievalPolicy?: NonNullable<UiMessage['retrieval_policy']> | null;
  loading?: boolean;
  highlightId?: number | null;
  onClose: () => void;
}

const SKIP_REASON_LABELS: Record<string, string> = {
  empty_query: 'Query was empty after trimming.',
  query_too_short: 'Query was too short (< 3 chars) to search.',
  greeting_only: 'Query was just a greeting, so no external search ran.',
  no_searchable_tokens: 'After stopword removal the query had no searchable terms.',
  all_below_relevance_floor:
    'Every retrieved candidate scored below the semantic-relevance floor for this query — nothing was on-topic enough to cite.',
};

function formatProviderName(name: string): string {
  const map: Record<string, string> = {
    openalex: 'OpenAlex',
    semanticscholar: 'Semantic Scholar',
    arxiv: 'arXiv',
    crossref: 'Crossref',
    springer: 'Springer',
    elsevier: 'Elsevier',
  };
  return map[name.toLowerCase()] ?? name;
}

export function SourcesPanel({
  mode,
  citations,
  trace,
  faithfulness,
  retrievalPolicy,
  loading,
  highlightId,
  onClose,
}: SourcesPanelProps) {
  const [citedOnly, setCitedOnly] = useState(false);
  const [expandedKey, setExpandedKey] = useState<string | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);

  // evidence_id -> list of claims that cited it. Used to tell the user which
  // sentence of the answer each chunk actually supports.
  const claimsByEvidence = useMemo(() => {
    const map = new Map<string, SupportingClaim[]>();
    for (const claim of faithfulness?.claims ?? []) {
      for (const evId of claim.evidence_ids ?? []) {
        if (!evId) continue;
        const list = map.get(evId) ?? [];
        list.push({ sentenceId: claim.sentence_id, sentence: claim.sentence });
        map.set(evId, list);
      }
    }
    return map;
  }, [faithfulness]);

  const groups = useMemo<SourceGroup[]>(() => {
    const traceById = new Map<number, WhyTraceChunk>();
    (trace || []).forEach((t, i) => traceById.set(t.id || i + 1, t));

    const byKey = new Map<string, SourceGroup>();
    const orderedKeys: string[] = [];

    (citations || []).forEach((c, i) => {
      const id = c.id || i + 1;
      const t = traceById.get(id);
      const key = sourceDedupKey({
        doc_id: c.doc_id,
        source: c.source,
        url: c.url,
        title: c.title,
      });
      const cited = c.used_in_answer ?? t?.cited ?? false;
      const page = c.page ?? t?.page;
      const snippet = c.snippet || t?.snippet_preview || '';
      const chunk: ChunkDetail = {
        citationId: id,
        page,
        snippet,
        evidenceId: c.evidence_id,
        rerankScore: c.rerank_score ?? t?.rerank_norm,
        simScore: c.sim_score ?? t?.sim_score,
        msaScore: c.msa?.msa_score,
        msaM: c.msa?.M,
        msaS: c.msa?.S,
        msaA: c.msa?.A,
        cited,
        metadataOnly: Boolean(c.metadata_only),
        msaSupported: c.msa_supported,
        claims: c.evidence_id ? claimsByEvidence.get(c.evidence_id) ?? [] : [],
      };

      const existing = byKey.get(key);
      if (!existing) {
        orderedKeys.push(key);
        byKey.set(key, {
          key,
          displayId: 0,
          title: c.title || t?.title || `Document ${c.doc_id ?? t?.doc_id ?? '?'}`,
          doc_id: c.doc_id ?? t?.doc_id,
          url: c.url,
          source: c.source,
          cited,
          confidence_obj: c.confidence_obj,
          pages: typeof page === 'number' ? [page] : [],
          chunks: [chunk],
          msaTop: chunk.msaScore,
          metadataOnly: chunk.metadataOnly,
        });
      } else {
        existing.cited = existing.cited || cited;
        if (typeof page === 'number' && !existing.pages.includes(page)) {
          existing.pages.push(page);
        }
        existing.chunks.push(chunk);
        if ((chunk.msaScore ?? 0) > (existing.msaTop ?? 0)) existing.msaTop = chunk.msaScore;
        if (!existing.confidence_obj && c.confidence_obj) existing.confidence_obj = c.confidence_obj;
      }
    });

    const result = orderedKeys.map((k) => byKey.get(k)!).filter(Boolean);
    result.sort((a, b) => Number(b.cited) - Number(a.cited));
    result.forEach((g, idx) => {
      g.displayId = idx + 1;
      g.pages.sort((a, b) => a - b);
      g.chunks.sort((a, b) => Number(b.cited) - Number(a.cited));
      // Group is metadata-only iff all chunks are metadata-only.
      g.metadataOnly = g.chunks.length > 0 && g.chunks.every((c) => c.metadataOnly);
      // Group is msaSupported iff any cited chunk has msaSupported=true.
      // Undefined (no MSA computed) stays undefined and renders no weak-support badge.
      const citedChunks = g.chunks.filter((c) => c.cited);
      if (citedChunks.length === 0) {
        g.msaSupported = undefined;
      } else if (citedChunks.every((c) => c.msaSupported === undefined)) {
        g.msaSupported = undefined;
      } else {
        g.msaSupported = citedChunks.some((c) => c.msaSupported === true);
      }
    });
    return result;
  }, [citations, trace, claimsByEvidence]);

  const visibleGroups = citedOnly ? groups.filter((g) => g.cited) : groups;
  const citedCount = groups.filter((g) => g.cited).length;

  useEffect(() => {
    if (highlightId == null) return;
    const target = groups.find((g) => g.chunks.some((c) => c.citationId === highlightId));
    if (target) {
      setExpandedKey(target.key);
      const el = containerRef.current?.querySelector<HTMLElement>(
        `[data-source-key="${target.key}"]`,
      );
      if (el) el.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
  }, [highlightId, groups]);

  const emptyState = useMemo(() => {
    if (loading || groups.length > 0) return null;

    const providerStatus = retrievalPolicy?.public_provider_status ?? {};
    const skippedEntry = Object.entries(providerStatus).find(([k]) => k === '_skipped');
    const skipped = skippedEntry ? (skippedEntry[1] as SkippedProviderEntry) : null;
    const realProviders = Object.entries(providerStatus).filter(
      ([k, v]) => k !== '_skipped' && !isSkippedEntry(v),
    ) as [string, ProviderStatusEntry][];
    const queriedProviders = realProviders.filter(([, v]) => v.queried);
    const abstentionReason =
      retrievalPolicy?.mode === 'abstention' ? retrievalPolicy.reason : undefined;

    if (mode !== 'public') {
      return {
        title: 'No evidence yet',
        hint: 'Send a message to see supporting excerpts here.',
        providers: [] as { name: string; fetched: number; selected: number }[],
      };
    }

    if (skipped) {
      return {
        title: 'No search ran for this query',
        hint: SKIP_REASON_LABELS[skipped.reason] ?? `Skipped: ${skipped.reason}`,
        detail: skipped.normalized_query
          ? `Normalized to: "${skipped.normalized_query}"`
          : undefined,
        providers: [] as { name: string; fetched: number; selected: number }[],
      };
    }

    if (queriedProviders.length > 0) {
      return {
        title: 'No sources matched',
        hint:
          abstentionReason === 'low-lexical-overlap'
            ? 'Retrieved candidates scored below the relevance floor for this query.'
            : abstentionReason
              ? `Abstained (${abstentionReason}).`
              : 'External providers returned no strong matches.',
        detail: 'Try a more specific phrase — a paper title, author, or concept keyword.',
        providers: queriedProviders.map(([name, meta]) => ({
          name: formatProviderName(name),
          fetched: meta.fetched ?? 0,
          selected: meta.selected ?? 0,
        })),
      };
    }

    return {
      title: 'No sources retrieved',
      hint: 'No external providers returned results for this query.',
      detail: 'Try a more specific phrase — a paper title, author, or concept keyword.',
      providers: [] as { name: string; fetched: number; selected: number }[],
    };
  }, [loading, groups.length, mode, retrievalPolicy]);

  return (
    <motion.aside
      initial={{ x: 40, opacity: 0 }}
      animate={{ x: 0, opacity: 1 }}
      exit={{ x: 40, opacity: 0 }}
      transition={{ duration: 0.25, ease: 'easeOut' }}
      className="flex h-full w-full max-w-[440px] flex-col border-l border-zinc-200 bg-white dark:border-zinc-800 dark:bg-zinc-950"
    >
      <div className="flex items-center justify-between gap-2 border-b border-zinc-200/70 px-4 py-3 dark:border-zinc-800/70">
        <div className="flex items-baseline gap-2">
          <span className="font-display text-lg leading-none text-zinc-900 dark:text-zinc-50">
            Evidence
          </span>
          {groups.length > 0 && (
            <span className="text-[11px] font-semibold tabular-nums text-zinc-500 dark:text-zinc-500">
              {citedCount}/{groups.length}
            </span>
          )}
        </div>
        <div className="flex items-center gap-0.5">
          <button
            type="button"
            className="rounded-lg p-1 text-zinc-500 hover:bg-zinc-100 dark:hover:bg-zinc-800"
            aria-label="Search evidence"
            title="Search — coming soon"
          >
            <Search size={14} />
          </button>
          <button
            type="button"
            onClick={onClose}
            className="rounded-lg p-1 text-zinc-500 hover:bg-zinc-100 dark:hover:bg-zinc-800"
            aria-label="Close"
          >
            <X size={16} />
          </button>
        </div>
      </div>
      {/* sub-header with column labels + loading hint */}
      <div className="flex items-center justify-between border-b border-zinc-200/70 px-4 py-1.5 text-[9.5px] font-semibold uppercase tracking-[0.16em] text-zinc-400 dark:border-zinc-800/70 dark:text-zinc-600">
        <span>
          {loading
            ? 'retrieving support…'
            : groups.length > 0
              ? `ID · SOURCE`
              : 'no evidence yet'}
        </span>
        {groups.length > 0 && <span>CONFIDENCE</span>}
      </div>

      {groups.length > 0 && citedCount !== groups.length && (
        <div className="flex gap-1 border-b border-zinc-200 px-4 py-2 dark:border-zinc-800">
          <button
            type="button"
            onClick={() => setCitedOnly(false)}
            className={cn(
              'rounded-full px-2.5 py-1 text-[11px] font-medium transition',
              !citedOnly
                ? 'bg-amber-500/15 text-amber-700 dark:text-amber-300'
                : 'text-zinc-500 hover:text-zinc-900 dark:hover:text-zinc-100',
            )}
          >
            All ({groups.length})
          </button>
          <button
            type="button"
            onClick={() => setCitedOnly(true)}
            className={cn(
              'rounded-full px-2.5 py-1 text-[11px] font-medium transition',
              citedOnly
                ? 'bg-amber-500/15 text-amber-700 dark:text-amber-300'
                : 'text-zinc-500 hover:text-zinc-900 dark:hover:text-zinc-100',
            )}
          >
            Cited ({citedCount})
          </button>
        </div>
      )}

      <div ref={containerRef} className="flex-1 overflow-y-auto px-4 py-3">
        {loading && groups.length === 0 ? (
          <div className="space-y-3">
            {[0, 1, 2].map((i) => (
              <div
                key={i}
                className="h-24 animate-pulse rounded-2xl border border-zinc-200 bg-zinc-100 dark:border-zinc-800 dark:bg-zinc-900"
              />
            ))}
          </div>
        ) : visibleGroups.length === 0 ? (
          emptyState ? (
            <div className="flex h-full flex-col items-center justify-center gap-3 text-center">
              <div className="flex h-10 w-10 items-center justify-center rounded-full bg-zinc-100 text-zinc-500 dark:bg-zinc-900 dark:text-zinc-400">
                <Search size={16} />
              </div>
              <div>
                <div className="text-sm font-semibold text-zinc-800 dark:text-zinc-100">
                  {emptyState.title}
                </div>
                <div className="mt-1 text-[11px] text-zinc-500 dark:text-zinc-400">
                  {emptyState.hint}
                </div>
                {emptyState.detail && (
                  <div className="mt-2 text-[11px] text-zinc-500 dark:text-zinc-400">
                    {emptyState.detail}
                  </div>
                )}
              </div>
              {emptyState.providers && emptyState.providers.length > 0 && (
                <div className="mt-2 w-full rounded-xl border border-zinc-200 bg-zinc-50/60 p-2 text-left dark:border-zinc-800 dark:bg-zinc-900/60">
                  <div className="mb-1.5 flex items-center gap-1 text-[10px] font-semibold uppercase tracking-wider text-zinc-500">
                    <Info size={10} /> Providers queried
                  </div>
                  <ul className="space-y-0.5">
                    {emptyState.providers.map((p) => (
                      <li
                        key={p.name}
                        className="flex items-center justify-between text-[11px]"
                      >
                        <span className="text-zinc-700 dark:text-zinc-300">{p.name}</span>
                        <span className="text-zinc-500">
                          {p.fetched > 0
                            ? `${p.fetched} fetched · ${p.selected} kept`
                            : 'no results'}
                        </span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          ) : null
        ) : (
          <div className="flex flex-col gap-2.5">
            {visibleGroups.map((g, idx) => {
              const isActive = highlightId != null
                && g.chunks.some((c) => c.citationId === highlightId);
              const isExpanded = expandedKey === g.key;
              const confScore = g.confidence_obj?.score;
              const confPct = confScore != null ? Math.round(confScore * 100) : null;
              const confLabel = g.confidence_obj?.label;
              const confTone =
                confPct == null
                  ? 'muted'
                  : confPct >= 60
                    ? 'support'
                    : confPct >= 40
                      ? 'accent'
                      : 'warn';
              const confTitle = (
                'Retrieval confidence: composite of semantic similarity, rerank score, ' +
                'citation coverage, and evidence margin.\n' +
                'Per-source score — how well retrieval found and placed this source for the query.'
              );
              const pageLabel =
                g.pages.length === 0
                  ? null
                  : g.pages.length === 1
                    ? `p.${g.pages[0]}`
                    : `pp.${g.pages.slice(0, 4).join(', ')}${g.pages.length > 4 ? '…' : ''}`;
              const excerptCount = g.chunks.filter(
                (c) => (c.snippet ?? '').trim().length > 0,
              ).length;
              const mappedCount = g.chunks.filter((c) => c.claims.length).length;

              return (
                <motion.div
                  key={g.key}
                  data-source-key={g.key}
                  data-source-id={g.displayId}
                  initial={{ opacity: 0, y: 6 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.18, delay: Math.min(idx * 0.02, 0.12) }}
                  className={cn(
                    'scroll-mt-4 rounded-2xl border p-3 text-left transition',
                    isActive
                      ? 'border-amber-500 bg-amber-500/10 ring-2 ring-amber-500/40'
                      : g.cited
                        ? 'border-amber-500/30 bg-amber-500/5'
                        : 'border-zinc-200 bg-white dark:border-zinc-800 dark:bg-zinc-900/60',
                  )}
                >
                  <div className="mb-1.5 flex flex-wrap items-center gap-1.5">
                    <span className="rounded-md bg-amber-500 px-1.5 py-0.5 font-mono text-[10px] font-semibold text-zinc-950">
                      S{g.displayId}
                    </span>
                    {g.source && <Badge tone="muted">{g.source}</Badge>}
                    {pageLabel && <Badge tone="muted">{pageLabel}</Badge>}
                    {confPct != null && (
                      <Badge tone={confTone} title={confTitle}>
                        Confidence {confLabel ?? ''} {confPct}%
                      </Badge>
                    )}
                    {g.metadataOnly && (
                      <Badge
                        tone="muted"
                        title="Source returned title + DOI but no abstract. Open the link for full text."
                      >
                        Metadata only
                      </Badge>
                    )}
                  </div>
                  {g.url ? (
                    <a
                      href={g.url}
                      target="_blank"
                      rel="noreferrer"
                      className="group flex items-start gap-1.5 text-sm font-medium text-zinc-900 hover:text-amber-600 hover:underline dark:text-zinc-50 dark:hover:text-amber-400"
                    >
                      <FileText size={12} className="mt-0.5 shrink-0 opacity-60" />
                      <span>{g.title}</span>
                      <ExternalLink
                        size={11}
                        className="mt-1 shrink-0 opacity-60 group-hover:opacity-100"
                      />
                    </a>
                  ) : (
                    <div className="flex items-start gap-1.5 text-sm font-medium text-zinc-900 dark:text-zinc-50">
                      <FileText size={12} className="mt-0.5 shrink-0 opacity-60" />
                      <span>{g.title}</span>
                    </div>
                  )}

                  <button
                    type="button"
                    onClick={() => setExpandedKey(isExpanded ? null : g.key)}
                    className="mt-2 flex w-full items-center justify-between rounded-lg border border-zinc-200 bg-white/60 px-2.5 py-1.5 text-[11px] font-medium text-zinc-600 transition hover:border-amber-500/40 hover:text-zinc-900 dark:border-zinc-800 dark:bg-zinc-950/40 dark:text-zinc-400 dark:hover:text-zinc-100"
                    aria-expanded={isExpanded}
                  >
                    <span>
                      {excerptCount === 0
                        ? 'Metadata only · no abstract'
                        : `${excerptCount} excerpt${excerptCount > 1 ? 's' : ''}`}
                      {faithfulness && mappedCount > 0 && (
                        <span className="ml-1 text-zinc-400">
                          · {mappedCount} mapped to claims
                        </span>
                      )}
                    </span>
                    <ChevronDown
                      size={12}
                      className={cn('transition', isExpanded && 'rotate-180')}
                    />
                  </button>

                  <AnimatePresence initial={false}>
                    {isExpanded && (
                      <motion.div
                        initial={{ opacity: 0, height: 0 }}
                        animate={{ opacity: 1, height: 'auto' }}
                        exit={{ opacity: 0, height: 0 }}
                        transition={{ duration: 0.18 }}
                        className="overflow-hidden"
                      >
                        <ul className="mt-2 flex flex-col gap-2">
                          {g.chunks.map((c, ci) => (
                            <li
                              key={`${c.citationId}-${ci}`}
                              className={cn(
                                'rounded-xl border px-2.5 py-2 text-[11.5px] leading-relaxed',
                                c.cited
                                  ? 'border-amber-500/30 bg-amber-500/[0.04]'
                                  : 'border-zinc-200 bg-zinc-50/60 dark:border-zinc-800 dark:bg-zinc-900/40',
                              )}
                            >
                              <div className="mb-1 flex flex-wrap items-center gap-1.5 text-[10px] text-zinc-500">
                                <span className="font-mono text-zinc-600 dark:text-zinc-300">
                                  #{ci + 1}
                                </span>
                                {typeof c.page === 'number' && (
                                  <span className="text-zinc-500">p.{c.page}</span>
                                )}
                              </div>
                              {c.snippet ? (
                                <div className="text-zinc-700 dark:text-zinc-200 line-clamp-4">
                                  {c.snippet}
                                </div>
                              ) : (
                                <div className="space-y-1">
                                  <div className="italic text-zinc-400">
                                    No abstract returned by the source.
                                  </div>
                                  {g.url && (
                                    <a
                                      href={g.url}
                                      target="_blank"
                                      rel="noreferrer"
                                      className="inline-flex items-center gap-1 text-[11px] font-medium text-amber-600 hover:underline dark:text-amber-400"
                                    >
                                      Open link for full text
                                      <ExternalLink size={10} />
                                    </a>
                                  )}
                                </div>
                              )}
                              {c.claims.length > 0 && (
                                <div className="mt-1.5 border-t border-zinc-200/70 pt-1.5 dark:border-zinc-800/70">
                                  <div className="mb-0.5 text-[10px] font-semibold uppercase tracking-wider text-zinc-500">
                                    Supports
                                  </div>
                                  <ul className="space-y-0.5">
                                    {c.claims.map((cl) => (
                                      <li
                                        key={cl.sentenceId}
                                        className="text-[11px] text-zinc-600 dark:text-zinc-300"
                                      >
                                        <span className="mr-1 font-mono text-[10px] text-amber-600 dark:text-amber-400">
                                          #{cl.sentenceId + 1}
                                        </span>
                                        {cl.sentence}
                                      </li>
                                    ))}
                                  </ul>
                                </div>
                              )}
                            </li>
                          ))}
                        </ul>
                      </motion.div>
                    )}
                  </AnimatePresence>

                  {mode === 'public' && g.doc_id != null && (
                    <div className="mt-2">
                      <Link
                        to={`/?doc=${g.doc_id}`}
                        className="inline-flex items-center gap-1 text-[11px] font-medium text-amber-600 hover:underline dark:text-amber-400"
                        title="Open in your uploaded library"
                      >
                        <FolderOpen size={11} />
                        Open in library
                      </Link>
                    </div>
                  )}
                </motion.div>
              );
            })}
          </div>
        )}
      </div>
      {/* confidence legend — anchors the panel visually */}
      {groups.length > 0 && (
        <div className="flex items-center justify-between gap-2 border-t border-zinc-200/70 px-4 py-2 text-[9.5px] font-semibold uppercase tracking-[0.14em] text-zinc-400 dark:border-zinc-800/70 dark:text-zinc-600">
          <span className="inline-flex items-center gap-1.5">
            <span className="h-1.5 w-1.5 rounded-full bg-emerald-500" />
            ≥60
          </span>
          <span className="inline-flex items-center gap-1.5">
            <span className="h-1.5 w-1.5 rounded-full bg-amber-500" />
            40–59
          </span>
          <span className="inline-flex items-center gap-1.5">
            <span className="h-1.5 w-1.5 rounded-full bg-red-500" />
            &lt;40
          </span>
          <span className="text-zinc-300 dark:text-zinc-700">confidence</span>
        </div>
      )}
    </motion.aside>
  );
}
