import { useCallback, useEffect, useMemo, useState } from 'react';
import { API_BASE, api } from '../api/client';
import type {
  Citation,
  ConfidenceObject,
  DocumentRow,
  FaithfulnessReport,
  WhyTraceChunk,
} from '../api/types';

export type UiMessage = {
  role: 'you' | 'assistant';
  text: string;
  streaming?: boolean;
  citations?: Citation[];
  confidence?: ConfidenceObject;
  why_answer?: { rerank_changed_order: boolean; top_chunks: WhyTraceChunk[] };
  latency_breakdown_ms?: { retrieve: number; rerank: number; generate: number; total: number };
  needs_clarification?: boolean;
  clarification?: { question: string; options: string[]; recommended_option?: string } | null;
  faithfulness?: FaithfulnessReport | null;
  answer_scope?: string;
  unsupported_claims?: number;
  query_ref?: string;
  retrieval_policy?: {
    mode?: string;
    reason?: string;
    public_provider_status?: Record<
      string,
      ProviderStatusEntry | SkippedProviderEntry
    >;
    query_intent?: {
      canonical_term?: string | null;
      domain?: string | null;
      is_ambiguous?: boolean;
      alternative_senses?: string[];
      disambiguation_hints?: string[];
      search_queries?: string[];
      model?: string;
    } | null;
  };
};

export type ProviderStatusEntry = {
  selected?: number;
  fetched?: number;
  contributed?: boolean;
  queried?: boolean;
  variant?: string | null;
  available?: boolean;
  reason?: string | null;
};

export type SkippedProviderEntry = {
  reason: string;
  normalized_query?: string;
};

export function isSkippedEntry(entry: unknown): entry is SkippedProviderEntry {
  return (
    typeof entry === 'object'
    && entry !== null
    && typeof (entry as { reason?: unknown }).reason === 'string'
    && !('queried' in (entry as object))
  );
}

const STORAGE_PREFIX = 'scholarrag_chat_v1';
const HISTORY_PREFIX = 'scholarrag_chats_v2';

function storageKey(mode: 'uploaded' | 'public'): string {
  return `${STORAGE_PREFIX}::${mode}`;
}

function historyKey(mode: 'uploaded' | 'public'): string {
  return `${HISTORY_PREFIX}::${mode}`;
}

export interface ChatThreadSummary {
  id: string;
  title: string;
  updatedAt: number;
  messageCount: number;
}

interface StoredThread {
  id: string;
  title: string;
  updatedAt: number;
  messages: UiMessage[];
  selectedDocs: number[];
}

interface StoredHistory {
  activeId: string | null;
  threads: StoredThread[];
}

function loadHistory(mode: 'uploaded' | 'public'): StoredHistory {
  try {
    const raw = localStorage.getItem(historyKey(mode));
    if (!raw) return { activeId: null, threads: [] };
    const parsed = JSON.parse(raw) as StoredHistory;
    if (!parsed || !Array.isArray(parsed.threads)) return { activeId: null, threads: [] };
    return parsed;
  } catch {
    return { activeId: null, threads: [] };
  }
}

function saveHistory(mode: 'uploaded' | 'public', hist: StoredHistory): void {
  try {
    localStorage.setItem(historyKey(mode), JSON.stringify(hist));
  } catch {
    /* ignore */
  }
}

function deriveTitle(messages: UiMessage[]): string {
  const first = messages.find((m) => m.role === 'you');
  if (!first) return 'New chat';
  const t = first.text.trim().replace(/\s+/g, ' ');
  return t.length > 60 ? `${t.slice(0, 57)}...` : t;
}

function newThreadId(): string {
  return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 7)}`;
}

export function sourceDedupKey(citation: Pick<Citation, 'doc_id' | 'source' | 'url' | 'title'>): string {
  return citation.doc_id
    ? `uploaded|${citation.doc_id}`
    : `${citation.source || ''}|${citation.url || ''}|${citation.title || ''}`;
}

export function uniqueSourceCount(citations?: Citation[]): number {
  const keys = new Set<string>();
  for (const citation of citations || []) {
    keys.add(sourceDedupKey(citation));
  }
  return keys.size;
}

interface UseChatSessionOptions {
  mode: 'uploaded' | 'public';
}

export function useChatSession({ mode }: UseChatSessionOptions) {
  const [messages, setMessages] = useState<UiMessage[]>([]);
  const [docs, setDocs] = useState<DocumentRow[]>([]);
  const [selectedDocs, setSelectedDocs] = useState<number[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [history, setHistoryState] = useState<StoredHistory>({ activeId: null, threads: [] });

  const activeThreadId = history.activeId;
  const chatList = useMemo<ChatThreadSummary[]>(
    () =>
      history.threads
        .slice()
        .sort((a, b) => b.updatedAt - a.updatedAt)
        .map((t) => ({
          id: t.id,
          title: t.title,
          updatedAt: t.updatedAt,
          messageCount: t.messages.length,
        })),
    [history.threads],
  );

  // Restore on mount: prefer the v2 multi-thread history; if absent but the
  // legacy v1 single-session blob exists, migrate it into a first thread so
  // users don't lose their last conversation.
  useEffect(() => {
    const stored = loadHistory(mode);
    if (stored.threads.length > 0) {
      setHistoryState(stored);
      const active =
        stored.threads.find((t) => t.id === stored.activeId) || stored.threads[0];
      setMessages(active.messages);
      setSelectedDocs(active.selectedDocs || []);
      return;
    }
    try {
      const legacy = localStorage.getItem(storageKey(mode));
      if (legacy) {
        const parsed = JSON.parse(legacy);
        const legacyMessages: UiMessage[] = Array.isArray(parsed?.messages) ? parsed.messages : [];
        const legacyDocs: number[] = Array.isArray(parsed?.selectedDocs) ? parsed.selectedDocs : [];
        if (legacyMessages.length > 0) {
          const id = newThreadId();
          const thread: StoredThread = {
            id,
            title: deriveTitle(legacyMessages),
            updatedAt: Date.now(),
            messages: legacyMessages,
            selectedDocs: legacyDocs,
          };
          const next: StoredHistory = { activeId: id, threads: [thread] };
          setHistoryState(next);
          saveHistory(mode, next);
          setMessages(legacyMessages);
          setSelectedDocs(legacyDocs);
          return;
        }
      }
    } catch {
      /* ignore */
    }
  }, [mode]);

  // Persist the active thread whenever it changes.
  useEffect(() => {
    setHistoryState((prev) => {
      if (!prev.activeId) {
        if (messages.length === 0) return prev;
        const id = newThreadId();
        const thread: StoredThread = {
          id,
          title: deriveTitle(messages),
          updatedAt: Date.now(),
          messages,
          selectedDocs,
        };
        const next = { activeId: id, threads: [thread, ...prev.threads] };
        saveHistory(mode, next);
        return next;
      }
      const idx = prev.threads.findIndex((t) => t.id === prev.activeId);
      if (idx < 0) return prev;
      const existing = prev.threads[idx];
      const updated: StoredThread = {
        ...existing,
        title: existing.title === 'New chat' || !existing.title
          ? deriveTitle(messages)
          : existing.title,
        messages,
        selectedDocs,
        updatedAt: Date.now(),
      };
      const threads = prev.threads.slice();
      threads[idx] = updated;
      const next = { ...prev, threads };
      saveHistory(mode, next);
      return next;
    });
    try {
      localStorage.setItem(
        storageKey(mode),
        JSON.stringify({ messages, selectedDocs }),
      );
    } catch {
      /* ignore */
    }
  }, [mode, messages, selectedDocs]);

  // Docs for uploaded mode
  const refreshDocs = useCallback(async () => {
    if (mode !== 'uploaded') return;
    try {
      const res = await api.listDocs(50);
      const list = res.documents || [];
      setDocs(list);
      setSelectedDocs((prev) => prev.filter((id) => list.some((d) => d.id === id)));
      setError('');
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : 'Unknown error';
      setError(msg || `Backend unreachable at ${API_BASE}`);
    }
  }, [mode]);

  useEffect(() => {
    if (mode === 'uploaded') refreshDocs();
  }, [mode, refreshDocs]);

  // Auto-select a doc if the user navigated in via `?doc=<id>` — used to
  // open an uploaded reference from public-mode evidence.
  useEffect(() => {
    if (mode !== 'uploaded') return;
    if (typeof window === 'undefined') return;
    const params = new URLSearchParams(window.location.search);
    const raw = params.get('doc');
    if (!raw) return;
    const id = Number.parseInt(raw, 10);
    if (!Number.isFinite(id)) return;
    setSelectedDocs((prev) => (prev.includes(id) ? prev : [...prev, id]));
    // Clean up the URL so a refresh doesn't keep re-selecting.
    params.delete('doc');
    const next = `${window.location.pathname}${params.toString() ? `?${params}` : ''}`;
    window.history.replaceState({}, '', next);
  }, [mode]);

  // Poll while any doc is processing
  useEffect(() => {
    if (mode !== 'uploaded') return;
    if (!docs.some((d) => d.status === 'processing')) return;
    const id = window.setInterval(refreshDocs, 2500);
    return () => window.clearInterval(id);
  }, [mode, docs, refreshDocs]);

  const ask = useCallback(
    async (raw: string, opts?: { skipClear?: boolean; sense?: string }) => {
      const q = raw.trim();
      if (!q) return;
      setError('');
      setLoading(true);
      setMessages((prev) => [...prev, { role: 'you', text: q }]);
      if (!opts?.skipClear) setInput('');
      try {
        const scope: 'uploaded' | 'public' = mode;
        const payload = {
          query: q,
          scope,
          sense: opts?.sense,
          k: 8,
          allow_general_background: scope === 'public',
          doc_id:
            scope === 'uploaded' && selectedDocs.length === 1 ? selectedDocs[0] : undefined,
          doc_ids:
            scope === 'uploaded' && selectedDocs.length > 1 ? selectedDocs : undefined,
        };
        const res = await api.askAssistant(payload);
        const text = (res.answer || res.clarification?.question || '').trim();
        const msg: UiMessage = {
          role: 'assistant',
          text: text || 'No response received. Check backend/OpenAI key.',
          citations: res.citations || [],
          confidence: res.confidence,
          why_answer: res.why_answer,
          latency_breakdown_ms: res.latency_breakdown_ms,
          needs_clarification: res.needs_clarification,
          clarification: res.clarification,
          answer_scope: res.answer_scope,
          unsupported_claims: res.unsupported_claims,
          faithfulness: res.faithfulness,
          query_ref: q,
          retrieval_policy: res.retrieval_policy,
        };
        setMessages((prev) => [...prev, msg]);
      } catch (e: unknown) {
        const msg = e instanceof Error ? e.message : `Backend unreachable at ${API_BASE}`;
        setError(msg);
        setMessages((prev) => [...prev, { role: 'assistant', text: msg, citations: [] }]);
      } finally {
        setLoading(false);
      }
    },
    [mode, selectedDocs],
  );

  const uploadFiles = useCallback(
    async (files: FileList | File[]) => {
      if (mode !== 'uploaded') return;
      const arr = Array.from(files);
      if (!arr.length) return;
      try {
        for (const f of arr) {
          await api.uploadFile(f);
        }
        await refreshDocs();
      } catch (e: unknown) {
        const msg = e instanceof Error ? e.message : 'Upload failed';
        setError(msg);
      }
    },
    [mode, refreshDocs],
  );

  const deleteDoc = useCallback(
    async (id: number) => {
      try {
        await api.deleteDoc(id);
        setSelectedDocs((prev) => prev.filter((x) => x !== id));
        await refreshDocs();
      } catch (e: unknown) {
        const msg = e instanceof Error ? e.message : 'Failed to delete';
        setError(msg);
      }
    },
    [refreshDocs],
  );

  const toggleDoc = useCallback((id: number) => {
    setSelectedDocs((prev) => (prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id]));
  }, []);

  const clearChat = useCallback(() => {
    setMessages([]);
    setError('');
    setInput('');
  }, []);

  const newChat = useCallback(() => {
    setMessages([]);
    setError('');
    setInput('');
    setHistoryState((prev) => {
      const next: StoredHistory = { ...prev, activeId: null };
      saveHistory(mode, next);
      return next;
    });
  }, [mode]);

  const selectChat = useCallback(
    (id: string) => {
      setHistoryState((prev) => {
        const thread = prev.threads.find((t) => t.id === id);
        if (!thread) return prev;
        setMessages(thread.messages);
        setSelectedDocs(thread.selectedDocs || []);
        setError('');
        const next: StoredHistory = { ...prev, activeId: id };
        saveHistory(mode, next);
        return next;
      });
    },
    [mode],
  );

  const deleteChat = useCallback(
    (id: string) => {
      setHistoryState((prev) => {
        const threads = prev.threads.filter((t) => t.id !== id);
        let activeId = prev.activeId;
        if (activeId === id) {
          activeId = threads[0]?.id ?? null;
          if (activeId) {
            const t = threads.find((x) => x.id === activeId)!;
            setMessages(t.messages);
            setSelectedDocs(t.selectedDocs || []);
          } else {
            setMessages([]);
            setSelectedDocs([]);
          }
        }
        const next: StoredHistory = { activeId, threads };
        saveHistory(mode, next);
        return next;
      });
    },
    [],
  );

  const renameChat = useCallback(
    (id: string, title: string) => {
      setHistoryState((prev) => {
        const threads = prev.threads.map((t) =>
          t.id === id ? { ...t, title: title.trim() || t.title, updatedAt: Date.now() } : t,
        );
        const next: StoredHistory = { ...prev, threads };
        saveHistory(mode, next);
        return next;
      });
    },
    [mode],
  );

  const dedupedDocs = useMemo(() => {
    const seen = new Set<string>();
    return docs.filter((d) => {
      const k = d.title.toLowerCase();
      if (seen.has(k)) return false;
      seen.add(k);
      return true;
    });
  }, [docs]);

  return {
    messages,
    setMessages,
    input,
    setInput,
    loading,
    error,
    docs: dedupedDocs,
    selectedDocs,
    setSelectedDocs,
    toggleDoc,
    ask,
    uploadFiles,
    deleteDoc,
    refreshDocs,
    clearChat,
    chatList,
    activeThreadId,
    newChat,
    selectChat,
    deleteChat,
    renameChat,
  };
}
