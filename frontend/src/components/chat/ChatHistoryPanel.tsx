import { motion } from 'framer-motion';
import { MessageSquarePlus, Trash2 } from 'lucide-react';
import type { ChatThreadSummary } from '../../hooks/useChatSession';
import { cn } from '../../lib/cn';

interface ChatHistoryPanelProps {
  chats: ChatThreadSummary[];
  activeId: string | null;
  onNew: () => void;
  onSelect: (id: string) => void;
  onDelete: (id: string) => void;
}

function formatRelative(ts: number): string {
  const diff = Date.now() - ts;
  const min = Math.round(diff / 60000);
  if (min < 1) return 'just now';
  if (min < 60) return `${min}m ago`;
  const hr = Math.round(min / 60);
  if (hr < 24) return `${hr}h ago`;
  const d = Math.round(hr / 24);
  if (d < 7) return `${d}d ago`;
  return new Date(ts).toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
}

export function ChatHistoryPanel({
  chats,
  activeId,
  onNew,
  onSelect,
  onDelete,
}: ChatHistoryPanelProps) {
  return (
    <div className="flex h-full w-full flex-col">
      <div className="border-b border-zinc-200 px-3 py-3 dark:border-zinc-800">
        <button
          type="button"
          onClick={onNew}
          className="flex w-full items-center justify-center gap-1.5 rounded-xl border border-amber-500/40 bg-amber-500/10 px-3 py-2 text-[12px] font-semibold text-amber-700 transition hover:bg-amber-500 hover:text-zinc-950 dark:text-amber-300 dark:hover:text-zinc-950"
        >
          <MessageSquarePlus size={13} />
          New chat
        </button>
      </div>
      <div className="flex-1 overflow-y-auto px-2 py-2">
        {chats.length === 0 ? (
          <div className="px-2 py-6 text-center text-[11px] text-zinc-500 dark:text-zinc-500">
            No past chats yet. Start a conversation to build history.
          </div>
        ) : (
          <div className="flex flex-col gap-1">
            {chats.map((c) => (
              <motion.button
                key={c.id}
                type="button"
                layout
                onClick={() => onSelect(c.id)}
                className={cn(
                  'group relative flex flex-col items-start gap-0.5 rounded-lg px-2.5 py-2 text-left transition',
                  activeId === c.id
                    ? 'bg-amber-500/15 text-amber-800 dark:text-amber-200'
                    : 'text-zinc-700 hover:bg-zinc-100 dark:text-zinc-300 dark:hover:bg-zinc-900',
                )}
              >
                <div className="flex w-full items-center justify-between gap-2">
                  <span className="truncate text-[12px] font-medium">{c.title || 'New chat'}</span>
                  <span
                    role="button"
                    tabIndex={0}
                    onClick={(e) => {
                      e.stopPropagation();
                      if (
                        typeof window !== 'undefined' &&
                        !window.confirm('Delete this chat?')
                      )
                        return;
                      onDelete(c.id);
                    }}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter' || e.key === ' ') {
                        e.preventDefault();
                        e.stopPropagation();
                        onDelete(c.id);
                      }
                    }}
                    className="shrink-0 rounded p-0.5 text-zinc-400 opacity-0 transition hover:bg-zinc-200 hover:text-rose-600 group-hover:opacity-100 dark:hover:bg-zinc-800"
                    aria-label="Delete chat"
                  >
                    <Trash2 size={11} />
                  </span>
                </div>
                <span className="text-[10px] text-zinc-500 dark:text-zinc-500">
                  {formatRelative(c.updatedAt)} · {c.messageCount} msg
                  {c.messageCount === 1 ? '' : 's'}
                </span>
              </motion.button>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
