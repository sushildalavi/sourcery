import { useEffect, useRef } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import { Sparkles } from 'lucide-react';
import type { UiMessage } from '../../hooks/useChatSession';
import { MessageBubble } from './MessageBubble';

interface MessageListProps {
  messages: UiMessage[];
  loading?: boolean;
  activeIdx?: number;
  onActivateIdx?: (idx: number) => void;
  onClarify?: (msg: UiMessage, option: string) => void;
  onCite?: (msgIdx: number, citeId: number) => void;
  onFollowUp?: (prompt: string) => void;
  onNewThread?: () => void;
  followUpSuggestions?: string[];
}

function TypingIndicator() {
  return (
    <motion.div
      className="flex items-center gap-2.5"
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -4 }}
      transition={{ duration: 0.2 }}
    >
      <div
        aria-hidden
        className="flex h-8 w-8 items-center justify-center rounded-xl bg-amber-500/15 text-amber-600 dark:text-amber-400"
      >
        <Sparkles size={14} />
      </div>
      <div className="flex items-center gap-1 rounded-2xl border border-zinc-200 bg-white px-4 py-3 dark:border-zinc-800 dark:bg-zinc-900">
        {[0, 1, 2].map((i) => (
          <motion.span
            key={i}
            className="h-1.5 w-1.5 rounded-full bg-zinc-400 dark:bg-zinc-500"
            animate={{ y: [0, -4, 0], opacity: [0.4, 1, 0.4] }}
            transition={{ duration: 1.1, repeat: Infinity, delay: i * 0.15 }}
          />
        ))}
      </div>
    </motion.div>
  );
}

export function MessageList({
  messages,
  loading,
  activeIdx,
  onActivateIdx,
  onClarify,
  onCite,
  onFollowUp,
  onNewThread,
  followUpSuggestions,
}: MessageListProps) {
  const endRef = useRef<HTMLDivElement | null>(null);
  const prevCountRef = useRef(0);

  useEffect(() => {
    const prev = prevCountRef.current;
    prevCountRef.current = messages.length;
    if (messages.length > prev && prev > 0) {
      endRef.current?.scrollIntoView({ behavior: 'smooth' });
    } else if (loading) {
      endRef.current?.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages, loading]);

  // Find the index of the latest settled assistant reply — only it gets follow-up chips.
  let lastAssistantIdx = -1;
  for (let i = messages.length - 1; i >= 0; i--) {
    const m = messages[i];
    if (m.role === 'assistant' && !m.streaming) {
      lastAssistantIdx = i;
      break;
    }
  }

  return (
    <div className="flex flex-col gap-5 pb-2">
      {messages.map((msg, idx) => (
        <MessageBubble
          key={idx}
          message={msg}
          active={activeIdx === idx}
          onActivate={msg.role === 'assistant' ? () => onActivateIdx?.(idx) : undefined}
          onClarify={msg.role === 'assistant' ? (opt) => onClarify?.(msg, opt) : undefined}
          onCite={msg.role === 'assistant' ? (id) => onCite?.(idx, id) : undefined}
          onFollowUp={idx === lastAssistantIdx && !loading ? onFollowUp : undefined}
          onNewThread={idx === lastAssistantIdx && !loading ? onNewThread : undefined}
          followUpSuggestions={idx === lastAssistantIdx && !loading ? followUpSuggestions : undefined}
        />
      ))}
      <AnimatePresence>{loading && <TypingIndicator />}</AnimatePresence>
      <div ref={endRef} />
    </div>
  );
}
