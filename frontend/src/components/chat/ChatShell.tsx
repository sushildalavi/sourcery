import { useEffect, useMemo, useState } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import { FileText, History, Library, MessageSquarePlus, PanelLeft, Trash2, X } from 'lucide-react';
import { useChatSession, type UiMessage } from '../../hooks/useChatSession';
import { Composer } from './Composer';
import { MessageList } from './MessageList';
import { HomeHero } from './HomeHero';
import { AmbientBackground } from './AmbientBackground';
import { SourcesPanel } from './SourcesPanel';
import { DocumentsPanel } from './DocumentsPanel';
import { ChatHistoryPanel } from './ChatHistoryPanel';
import {
  CommandPalette,
  MessageSquarePlus as PaletteNewIcon,
  Sparkles as PaletteEvidenceIcon,
  type PaletteAction,
} from './CommandPalette';
import { useDefaultPaletteActions } from './usePaletteActions';
import { Badge } from '../ui/Badge';
import { Button } from '../ui/Button';
import { cn } from '../../lib/cn';

interface ChatShellProps {
  mode: 'uploaded' | 'public';
  title: string;
  description: string;
  emptyPrompts: {
    eyebrow: string;
    title: string;
    description: string;
    prompt: string;
  }[];
}

export function ChatShell({ mode, title, description, emptyPrompts }: ChatShellProps) {
  const {
    messages,
    input,
    setInput,
    loading,
    error,
    docs,
    selectedDocs,
    toggleDoc,
    ask,
    uploadFiles,
    deleteDoc,
    clearChat,
    chatList,
    activeThreadId,
    newChat,
    selectChat,
    deleteChat,
  } = useChatSession({ mode });

  const navigate = useNavigate();
  const [activeEvidenceIdx, setActiveEvidenceIdx] = useState<number | null>(null);
  const [sourcesOpen, setSourcesOpen] = useState(true);
  const [leftTab, setLeftTab] = useState<'history' | 'docs'>(mode === 'uploaded' ? 'docs' : 'history');
  const [leftPanelOpen, setLeftPanelOpen] = useState(true);
  const [highlightCiteId, setHighlightCiteId] = useState<number | null>(null);

  // Reset evidence + switch to the most natural left tab when the active
  // thread changes — new chats should feel like a fresh surface.
  useEffect(() => {
    setActiveEvidenceIdx(null);
    setHighlightCiteId(null);
  }, [activeThreadId]);

  // Auto-follow the latest settled assistant reply so the evidence panel
  // stays current. Includes replies with zero citations — the panel now
  // surfaces an explanatory empty state (especially useful for public-mode
  // abstentions) instead of disappearing entirely.
  useEffect(() => {
    for (let i = messages.length - 1; i >= 0; i--) {
      const m = messages[i];
      if (m.role === 'assistant' && !m.streaming) {
        setActiveEvidenceIdx(i);
        setSourcesOpen(true);
        return;
      }
    }
  }, [messages]);

  const activeEvidence = useMemo(() => {
    if (activeEvidenceIdx == null) return null;
    const msg = messages[activeEvidenceIdx];
    if (!msg || msg.role !== 'assistant') return null;
    return {
      citations: msg.citations || [],
      trace: msg.why_answer?.top_chunks || [],
      faithfulness: msg.faithfulness ?? null,
      retrievalPolicy: msg.retrieval_policy ?? null,
    };
  }, [activeEvidenceIdx, messages]);

  const handleActivate = (idx: number) => {
    setActiveEvidenceIdx(idx);
    setSourcesOpen(true);
  };

  const handleCite = (idx: number, citeId: number) => {
    setActiveEvidenceIdx(idx);
    setSourcesOpen(true);
    setHighlightCiteId(citeId);
    // Re-trigger scroll even if the id hasn't changed.
    window.setTimeout(() => setHighlightCiteId((v) => (v === citeId ? citeId : v)), 0);
  };

  const handleClarify = (msg: UiMessage, option: string) => {
    if (!msg.query_ref) return;
    void ask(msg.query_ref, { sense: option });
  };

  const handleSend = () => {
    if (!input.trim() || loading) return;
    void ask(input);
  };

  const handlePromptSelect = (prompt: string) => {
    if (loading) return;
    void ask(prompt);
  };

  // Derive follow-up suggestions from the latest settled reply. For now we
  // hand-craft a small bank of useful next steps; in a future revision these
  // could come from the backend (e.g. GPT-generated per-answer prompts).
  const followUpSuggestions = useMemo<string[]>(() => {
    const lastAssistant = [...messages].reverse().find((m) => m.role === 'assistant' && !m.streaming);
    if (!lastAssistant) return [];
    const qi = lastAssistant.retrieval_policy?.query_intent;
    const canonical = qi?.canonical_term;
    if (canonical) {
      return [
        `How does ${canonical} compare to alternatives?`,
        `What are the limitations of ${canonical}?`,
        `Show me the seminal paper on ${canonical}.`,
      ];
    }
    if (mode === 'public') {
      return [
        'What are the most-cited papers?',
        'Summarise the methodology.',
        'How does this compare to prior work?',
      ];
    }
    return [
      'Summarise the key contributions.',
      'What are the main limitations?',
      'Compare this to related work.',
    ];
  }, [messages, mode]);

  const extraPaletteActions = useMemo<PaletteAction[]>(
    () => [
      {
        id: 'chat-new',
        label: 'Start a new conversation',
        description: 'Clear messages and evidence for this mode',
        icon: PaletteNewIcon,
        keywords: ['new', 'thread', 'chat', 'clear'],
        run: () => {
          newChat();
          setActiveEvidenceIdx(null);
          setHighlightCiteId(null);
        },
      },
      {
        id: 'chat-evidence',
        label: sourcesOpen ? 'Hide evidence panel' : 'Show evidence panel',
        description: 'Toggle the per-source evidence sidebar',
        icon: PaletteEvidenceIcon,
        keywords: ['evidence', 'sources', 'panel', 'sidebar'],
        run: () => setSourcesOpen((v) => !v),
      },
    ],
    [newChat, sourcesOpen],
  );
  const paletteActions = useDefaultPaletteActions(extraPaletteActions);

  const contextHint = useMemo(() => {
    if (mode === 'public') return 'Public research mode · Semantic Scholar, OpenAlex, arXiv, Crossref';
    if (selectedDocs.length > 0) {
      return `${selectedDocs.length} document${selectedDocs.length > 1 ? 's' : ''} selected`;
    }
    if (docs.length === 0) return 'Upload a document to get started';
    return 'Select a document from the panel to ground answers';
  }, [mode, selectedDocs, docs]);

  const placeholder =
    mode === 'public'
      ? 'Ask about the open literature...'
      : selectedDocs.length
        ? 'Ask about your selected documents...'
        : 'Select a document to begin...';

  return (
    <div className="flex h-full min-h-0 w-full">
      <CommandPalette actions={paletteActions} />
      {/* Left panel — tabs for History and (uploaded mode) Documents */}
      <AnimatePresence>
        {leftPanelOpen && (
          <motion.div
            initial={{ width: 0, opacity: 0 }}
            animate={{ width: 272, opacity: 1 }}
            exit={{ width: 0, opacity: 0 }}
            transition={{ duration: 0.22, ease: 'easeOut' }}
            className="hidden md:block shrink-0 overflow-hidden border-r border-zinc-200 bg-zinc-50/60 dark:border-zinc-800 dark:bg-zinc-900/40"
          >
            <div className="flex h-full w-[272px] flex-col">
              {mode === 'uploaded' && (
                <div className="flex shrink-0 items-center gap-1 border-b border-zinc-200 px-2 pt-2 dark:border-zinc-800">
                  <button
                    type="button"
                    onClick={() => setLeftTab('history')}
                    className={cn(
                      'flex flex-1 items-center justify-center gap-1.5 rounded-t-md px-2 py-1.5 text-[11px] font-semibold transition',
                      leftTab === 'history'
                        ? 'bg-white text-zinc-900 shadow-sm dark:bg-zinc-950 dark:text-zinc-50'
                        : 'text-zinc-500 hover:text-zinc-900 dark:hover:text-zinc-100',
                    )}
                  >
                    <History size={12} />
                    Chats
                  </button>
                  <button
                    type="button"
                    onClick={() => setLeftTab('docs')}
                    className={cn(
                      'flex flex-1 items-center justify-center gap-1.5 rounded-t-md px-2 py-1.5 text-[11px] font-semibold transition',
                      leftTab === 'docs'
                        ? 'bg-white text-zinc-900 shadow-sm dark:bg-zinc-950 dark:text-zinc-50'
                        : 'text-zinc-500 hover:text-zinc-900 dark:hover:text-zinc-100',
                    )}
                  >
                    <Library size={12} />
                    Docs
                  </button>
                </div>
              )}
              <div className="min-h-0 flex-1">
                {leftTab === 'history' || mode === 'public' ? (
                  <ChatHistoryPanel
                    chats={chatList}
                    activeId={activeThreadId}
                    onNew={() => {
                      newChat();
                      setActiveEvidenceIdx(null);
                      setHighlightCiteId(null);
                    }}
                    onSelect={(id) => {
                      selectChat(id);
                    }}
                    onDelete={deleteChat}
                  />
                ) : (
                  <DocumentsPanel
                    docs={docs}
                    selectedDocs={selectedDocs}
                    onToggle={toggleDoc}
                    onUpload={uploadFiles}
                    onDelete={deleteDoc}
                  />
                )}
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Center chat */}
      <div className="relative flex min-w-0 flex-1 flex-col">
        <AmbientBackground />
        {/* Top meta bar */}
        <div className="relative z-10 flex items-center justify-between gap-3 border-b border-zinc-200/70 bg-white/60 px-4 py-2.5 backdrop-blur-md dark:border-zinc-800/70 dark:bg-zinc-950/50">
          <div className="flex min-w-0 items-center gap-2">
            <button
              type="button"
              onClick={() => setLeftPanelOpen((v) => !v)}
              className="hidden md:inline-flex h-8 w-8 items-center justify-center rounded-lg text-zinc-500 hover:bg-zinc-100 hover:text-zinc-800 dark:hover:bg-zinc-800 dark:hover:text-zinc-100"
              aria-label="Toggle side panel"
            >
              <PanelLeft size={15} />
            </button>
            <button
              type="button"
              onClick={() => {
                newChat();
                setActiveEvidenceIdx(null);
                setHighlightCiteId(null);
              }}
              className="inline-flex h-8 items-center gap-1 rounded-lg px-2 text-[11px] font-semibold text-zinc-500 hover:bg-zinc-100 hover:text-zinc-900 dark:hover:bg-zinc-800 dark:hover:text-zinc-100"
              title="Start a new chat"
            >
              <MessageSquarePlus size={13} />
              <span className="hidden sm:inline">New</span>
            </button>
            <div className="flex min-w-0 flex-col leading-tight">
              <span className="truncate text-sm font-semibold text-zinc-900 dark:text-zinc-50">
                {title}
              </span>
              <span className="truncate text-[11px] text-zinc-500 dark:text-zinc-400">
                {contextHint}
              </span>
            </div>
          </div>

          {/* scope toggle: Public research ↔ My documents */}
          <div className="hidden shrink-0 items-center rounded-full border border-zinc-200/70 bg-white/60 p-0.5 backdrop-blur dark:border-zinc-800/80 dark:bg-zinc-900/50 md:inline-flex">
            <button
              type="button"
              onClick={() => {
                if (mode !== 'public') navigate('/public');
              }}
              className={cn(
                'inline-flex items-center gap-1.5 rounded-full px-3 py-1 text-[11px] font-semibold transition',
                mode === 'public'
                  ? 'bg-amber-500/15 text-amber-700 shadow-sm ring-1 ring-amber-500/30 dark:text-amber-300'
                  : 'text-zinc-500 hover:text-zinc-900 dark:hover:text-zinc-100',
              )}
            >
              <span className="h-1.5 w-1.5 rounded-full bg-amber-500" />
              Public research
            </button>
            <button
              type="button"
              onClick={() => {
                if (mode !== 'uploaded') navigate('/');
              }}
              className={cn(
                'inline-flex items-center gap-1.5 rounded-full px-3 py-1 text-[11px] font-semibold transition',
                mode === 'uploaded'
                  ? 'bg-amber-500/15 text-amber-700 shadow-sm ring-1 ring-amber-500/30 dark:text-amber-300'
                  : 'text-zinc-500 hover:text-zinc-900 dark:hover:text-zinc-100',
              )}
            >
              <FileText size={11} />
              My documents
            </button>
          </div>

          <div className="flex items-center gap-1.5">
            {mode === 'uploaded' && selectedDocs.length > 0 && (
              <Badge tone="accent">
                <Library size={11} />
                {selectedDocs.length} active
              </Badge>
            )}
            {/* sources-meter: compact summary of evidence count for the active reply */}
            {activeEvidence && activeEvidence.citations.length > 0 && (
              <button
                type="button"
                onClick={() => setSourcesOpen((v) => !v)}
                className="hidden items-center gap-2 rounded-full border border-zinc-200/70 bg-white/60 px-2.5 py-1 text-[11px] font-semibold text-zinc-600 transition hover:border-amber-500/40 hover:text-zinc-900 dark:border-zinc-800/70 dark:bg-zinc-900/50 dark:text-zinc-300 dark:hover:text-zinc-100 md:inline-flex"
                title={sourcesOpen ? 'Hide evidence panel' : 'Show evidence panel'}
              >
                <span className="relative inline-flex h-4 w-4 items-center justify-center">
                  <svg viewBox="0 0 16 16" className="absolute h-4 w-4 -rotate-90">
                    <circle
                      cx="8"
                      cy="8"
                      r="6"
                      fill="none"
                      stroke="currentColor"
                      strokeOpacity="0.18"
                      strokeWidth="2"
                    />
                    <circle
                      cx="8"
                      cy="8"
                      r="6"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="2"
                      strokeLinecap="round"
                      strokeDasharray={`${(Math.max(0, Math.min(1, (activeEvidence.citations.filter((c) => c.used_in_answer).length) / Math.max(1, activeEvidence.citations.length))) * 37.7).toFixed(1)} 37.7`}
                      className="text-amber-500"
                    />
                  </svg>
                </span>
                {activeEvidence.citations.length} SOURCES
              </button>
            )}
            {activeEvidence && activeEvidence.citations.length > 0 && (
              <Button variant="ghost" size="sm" onClick={() => setSourcesOpen((v) => !v)}>
                {sourcesOpen ? 'Hide evidence' : 'Show evidence'}
              </Button>
            )}
            {messages.length > 0 && (
              <Button
                variant="ghost"
                size="sm"
                onClick={() => {
                  clearChat();
                  setActiveEvidenceIdx(null);
                  setSourcesOpen(false);
                }}
              >
                <Trash2 size={13} />
                <span className="hidden sm:inline">Clear</span>
              </Button>
            )}
          </div>
        </div>

        {/* Messages */}
        <div className="relative flex-1 overflow-y-auto">
          <div className="mx-auto flex min-h-full w-full max-w-3xl flex-col gap-6 px-4 py-6">
            {messages.length === 0 && !loading && (
              <HomeHero
                mode={mode}
                title={title}
                description={description}
                prompts={emptyPrompts}
                onSelectPrompt={handlePromptSelect}
                paperCount={mode === 'uploaded' ? docs.length : 15}
              />
            )}

            {error && (
              <div className="rounded-2xl border border-red-500/30 bg-red-500/5 px-4 py-3 text-xs text-red-600 dark:text-red-300">
                {error}
              </div>
            )}

            <MessageList
              messages={messages}
              loading={loading}
              activeIdx={activeEvidenceIdx ?? undefined}
              onActivateIdx={handleActivate}
              onClarify={handleClarify}
              onCite={handleCite}
              onFollowUp={handlePromptSelect}
              onNewThread={() => {
                newChat();
                setActiveEvidenceIdx(null);
                setHighlightCiteId(null);
              }}
              followUpSuggestions={followUpSuggestions}
            />
          </div>
        </div>

        {/* Composer */}
        <div className="relative z-10 border-t border-zinc-200/70 bg-white/70 px-4 py-3 backdrop-blur-md dark:border-zinc-800/70 dark:bg-zinc-950/60">
          <div className="mx-auto w-full max-w-3xl">
            <Composer
              value={input}
              onChange={setInput}
              onSend={handleSend}
              onAttach={mode === 'uploaded' ? uploadFiles : undefined}
              placeholder={placeholder}
              contextHint={mode === 'uploaded' && selectedDocs.length === 0 && docs.length > 0
                ? 'Docs-only mode: select a document to ground answers'
                : undefined}
              isLoading={loading}
            />
            {/* providers / palette hint under the composer */}
            <div className="mt-2 flex items-center justify-between gap-2 px-1 text-[10px] font-medium uppercase tracking-[0.14em] text-zinc-400 dark:text-zinc-600">
              <div className="flex items-center gap-1.5">
                <span className="relative flex h-1.5 w-1.5 items-center justify-center">
                  <span className="absolute h-1.5 w-1.5 animate-ping rounded-full bg-amber-400/70" />
                  <span className="relative h-1 w-1 rounded-full bg-amber-500" />
                </span>
                {mode === 'public'
                  ? 'public research mode · s2 · openalex · arxiv · crossref'
                  : `uploaded documents mode · ${docs.length} paper${docs.length === 1 ? '' : 's'} indexed`}
              </div>
              <span className="hidden items-center gap-1 sm:inline-flex">
                <span className="rounded border border-zinc-200 bg-zinc-100 px-1 py-0 font-mono text-[9px] normal-case text-zinc-500 dark:border-zinc-800 dark:bg-zinc-900 dark:text-zinc-400">
                  ⌘K
                </span>
                palette
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Right sources panel — visible at md+ whenever there's an active
          assistant reply. Empty-citation replies still render the panel so the
          user sees *why* there are no sources (public-mode abstentions, etc.). */}
      <AnimatePresence>
        {sourcesOpen && activeEvidence && (
          <div className="hidden md:flex">
            <SourcesPanel
              mode={mode}
              citations={activeEvidence.citations}
              trace={activeEvidence.trace}
              faithfulness={activeEvidence.faithfulness}
              retrievalPolicy={activeEvidence.retrievalPolicy}
              loading={loading}
              highlightId={highlightCiteId}
              onClose={() => setSourcesOpen(false)}
            />
          </div>
        )}
      </AnimatePresence>

      {/* Mobile left drawer — hosts both Chats and Docs tabs */}
      <AnimatePresence>
        {leftPanelOpen && (
          <div className="md:hidden fixed inset-0 z-40 flex">
            <div
              className="absolute inset-0 bg-black/40 backdrop-blur-sm"
              onClick={() => setLeftPanelOpen(false)}
              aria-hidden
            />
            <motion.div
              initial={{ x: -300 }}
              animate={{ x: 0 }}
              exit={{ x: -300 }}
              transition={{ type: 'spring', stiffness: 420, damping: 36 }}
              className={cn(
                'relative h-full w-[84%] max-w-[300px] border-r border-zinc-200 bg-white',
                'dark:border-zinc-800 dark:bg-zinc-950',
              )}
            >
              <div className="flex items-center justify-between border-b border-zinc-200 px-3 py-2 dark:border-zinc-800">
                <span className="text-sm font-semibold">
                  {mode === 'public' || leftTab === 'history' ? 'Chats' : 'Documents'}
                </span>
                <button
                  type="button"
                  onClick={() => setLeftPanelOpen(false)}
                  className="rounded-lg p-1 text-zinc-500 hover:bg-zinc-100 dark:hover:bg-zinc-800"
                  aria-label="Close"
                >
                  <X size={15} />
                </button>
              </div>
              {mode === 'uploaded' && (
                <div className="flex items-center gap-1 border-b border-zinc-200 px-2 py-1.5 dark:border-zinc-800">
                  <button
                    type="button"
                    onClick={() => setLeftTab('history')}
                    className={cn(
                      'flex flex-1 items-center justify-center gap-1.5 rounded-md px-2 py-1.5 text-[11px] font-semibold transition',
                      leftTab === 'history'
                        ? 'bg-zinc-100 text-zinc-900 dark:bg-zinc-800 dark:text-zinc-50'
                        : 'text-zinc-500',
                    )}
                  >
                    <History size={12} /> Chats
                  </button>
                  <button
                    type="button"
                    onClick={() => setLeftTab('docs')}
                    className={cn(
                      'flex flex-1 items-center justify-center gap-1.5 rounded-md px-2 py-1.5 text-[11px] font-semibold transition',
                      leftTab === 'docs'
                        ? 'bg-zinc-100 text-zinc-900 dark:bg-zinc-800 dark:text-zinc-50'
                        : 'text-zinc-500',
                    )}
                  >
                    <Library size={12} /> Docs
                  </button>
                </div>
              )}
              <div className="h-full">
                {leftTab === 'history' || mode === 'public' ? (
                  <ChatHistoryPanel
                    chats={chatList}
                    activeId={activeThreadId}
                    onNew={() => {
                      newChat();
                      setLeftPanelOpen(false);
                    }}
                    onSelect={(id) => {
                      selectChat(id);
                      setLeftPanelOpen(false);
                    }}
                    onDelete={deleteChat}
                  />
                ) : (
                  <DocumentsPanel
                    docs={docs}
                    selectedDocs={selectedDocs}
                    onToggle={toggleDoc}
                    onUpload={uploadFiles}
                    onDelete={deleteDoc}
                  />
                )}
              </div>
            </motion.div>
          </div>
        )}
      </AnimatePresence>
    </div>
  );
}
