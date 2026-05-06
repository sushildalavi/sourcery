import { motion } from 'framer-motion';
import { Brain, Sparkles } from 'lucide-react';
import type { UiMessage } from '../../hooks/useChatSession';
import { uniqueSourceCount } from '../../hooks/useChatSession';
import { renderMarkdown } from './markdown';
import { ConfidenceBadge } from './ConfidenceBadge';
import { FollowUpChips } from './FollowUpChips';
import { Badge } from '../ui/Badge';
import { cn } from '../../lib/cn';

function formatAnswerScope(scope?: string): string | null {
  if (!scope) return null;
  const normalized = scope.trim().toLowerCase();
  const explicit: Record<string, string> = {
    official_document_context: 'Official document',
    uploaded_document_context: 'Uploaded document',
    personal_document_context: 'Personal document',
    public_research_context: 'Public research',
    mixed_research_context: 'Mixed research',
    context_limited: 'Context limited',
  };
  if (explicit[normalized]) return explicit[normalized];
  return normalized
    .replace(/_context$/, '')
    .replace(/_/g, ' ')
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

interface MessageBubbleProps {
  message: UiMessage;
  active?: boolean;
  onActivate?: () => void;
  onClarify?: (option: string) => void;
  onCite?: (id: number) => void;
  onFollowUp?: (prompt: string) => void;
  onNewThread?: () => void;
  followUpSuggestions?: string[];
}

export function MessageBubble({
  message,
  active,
  onActivate,
  onClarify,
  onCite,
  onFollowUp,
  onNewThread,
  followUpSuggestions,
}: MessageBubbleProps) {
  const isUser = message.role === 'you';
  const scopeLabel = formatAnswerScope(message.answer_scope);
  const citedCount = uniqueSourceCount(message.citations);

  if (isUser) {
    return (
      <motion.div
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.18, ease: 'easeOut' }}
        className="flex justify-end"
      >
        <div className="max-w-[85%] rounded-2xl bg-amber-500 px-4 py-2 text-[0.93rem] font-medium leading-relaxed text-zinc-950 shadow-sm">
          {message.text}
        </div>
      </motion.div>
    );
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.22, ease: 'easeOut' }}
      className="flex flex-col gap-2"
    >
      <div
        onClick={onActivate}
        role={onActivate ? 'button' : undefined}
        tabIndex={onActivate ? 0 : undefined}
        className={cn(
          'group relative rounded-2xl border border-zinc-200/70 bg-white/80 px-5 py-4 shadow-sm transition backdrop-blur-sm dark:border-zinc-800/70 dark:bg-zinc-900/60',
          onActivate && 'cursor-pointer hover:border-amber-500/40 dark:hover:border-amber-500/40',
          active && 'border-amber-500/50 ring-1 ring-amber-500/40 dark:border-amber-500/40',
        )}
      >
        {/* amber quote-bar on the left */}
        <span
          aria-hidden
          className="absolute inset-y-3 left-0 w-0.5 rounded-full bg-gradient-to-b from-amber-500/80 via-amber-500/40 to-transparent"
        />
        {/* ANSWER eyebrow */}
        <div className="mb-2 flex items-center gap-1.5 text-[10px] font-semibold uppercase tracking-[0.16em] text-amber-600 dark:text-amber-400">
          <Sparkles size={10} />
          answer
        </div>
        <div className="text-[0.93rem] leading-relaxed text-zinc-800 dark:text-zinc-100">
          {renderMarkdown(message.text, { onCite, citations: message.citations })}
          {message.streaming && <span className="stream-cursor" />}
        </div>
      </div>

      {!message.streaming && (
        <div className="flex flex-wrap items-center gap-x-2.5 gap-y-1 px-1 text-[11px] text-zinc-500 dark:text-zinc-400">
          <ConfidenceBadge confidence={message.confidence} showWhenMissing={Boolean(citedCount)} />
          {message.faithfulness && (
            <Badge tone="support">
              Faithful {Math.round((message.faithfulness.overall_score || 0) * 100)}%
            </Badge>
          )}
          {citedCount > 0 && (
            <span className="inline-flex items-center gap-1">
              <span className="h-1 w-1 rounded-full bg-zinc-400" />
              {citedCount} source{citedCount > 1 ? 's' : ''}
            </span>
          )}
          {scopeLabel && (
            <span className="inline-flex items-center gap-1">
              <span className="h-1 w-1 rounded-full bg-zinc-400" />
              {scopeLabel}
            </span>
          )}
          {(() => {
            const qi = message.retrieval_policy?.query_intent;
            if (!qi || !qi.canonical_term) return null;
            const domainPart = qi.domain ? ` \u00b7 ${qi.domain}` : '';
            const model = qi.model ?? 'gpt-4o-mini';
            const alt = qi.alternative_senses?.length
              ? `\nAlternatives: ${qi.alternative_senses.join(', ')}`
              : '';
            return (
              <span
                className="inline-flex items-center gap-1 text-amber-600 dark:text-amber-400"
                title={`Resolved by ${model}${alt}`}
              >
                <span className="h-1 w-1 rounded-full bg-zinc-400" />
                <Brain size={10} />
                resolved as <span className="font-medium">{qi.canonical_term}</span>
                {domainPart && <span className="text-zinc-500">{domainPart}</span>}
              </span>
            );
          })()}
        </div>
      )}

      {!message.streaming && onFollowUp && followUpSuggestions && followUpSuggestions.length > 0 && (
        <FollowUpChips
          suggestions={followUpSuggestions}
          onSelect={onFollowUp}
          onNewThread={onNewThread}
        />
      )}

      {!message.streaming && message.needs_clarification && message.clarification?.options?.length ? (
        <div className="mt-1 flex flex-col gap-2 rounded-2xl border border-amber-500/30 bg-amber-500/5 p-3">
          <div className="text-xs font-medium text-amber-700 dark:text-amber-300">
            {message.clarification.question}
          </div>
          <div className="flex flex-wrap gap-1.5">
            {message.clarification.options.map((opt) => (
              <button
                key={opt}
                type="button"
                onClick={() => onClarify?.(opt)}
                className="rounded-full border border-amber-500/40 bg-white px-3 py-1 text-xs font-medium text-amber-700 hover:bg-amber-500/10 dark:bg-zinc-900 dark:text-amber-300"
              >
                {opt}
              </button>
            ))}
          </div>
        </div>
      ) : null}
    </motion.div>
  );
}
