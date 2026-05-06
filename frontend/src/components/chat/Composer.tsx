import { useEffect, useRef, type KeyboardEvent } from 'react';
import { motion } from 'framer-motion';
import { Loader, Paperclip, Send } from 'lucide-react';
import { cn } from '../../lib/cn';

interface ComposerProps {
  value: string;
  onChange: (value: string) => void;
  onSend: () => void;
  onAttach?: (files: FileList | null) => void;
  placeholder?: string;
  contextHint?: string;
  disabled?: boolean;
  isLoading?: boolean;
}

export function Composer({
  value,
  onChange,
  onSend,
  onAttach,
  placeholder = 'Ask a research question...',
  contextHint,
  disabled,
  isLoading,
}: ComposerProps) {
  const ref = useRef<HTMLTextAreaElement | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    el.style.height = 'auto';
    el.style.height = `${Math.min(el.scrollHeight, 200)}px`;
  }, [value]);

  const canSend = value.trim().length > 0 && !disabled && !isLoading;

  const submit = () => {
    if (!canSend) return;
    onSend();
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      submit();
    }
  };

  return (
    <div className="w-full">
      <div className="rounded-2xl border border-zinc-200 bg-white shadow-sm transition focus-within:border-amber-500/60 focus-within:shadow-[0_0_0_3px_rgba(245,158,11,0.12)] dark:border-zinc-800 dark:bg-zinc-900">
        {contextHint && (
          <div className="border-b border-zinc-100 px-4 pt-3 pb-1.5 text-[11px] font-medium uppercase tracking-wider text-zinc-500 dark:border-zinc-800 dark:text-zinc-400">
            {contextHint}
          </div>
        )}
        <textarea
          ref={ref}
          value={value}
          onChange={(e) => onChange(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={placeholder}
          disabled={disabled || isLoading}
          rows={1}
          className="block w-full resize-none bg-transparent px-4 pt-3 pb-2 text-sm leading-relaxed outline-none placeholder:text-zinc-400 disabled:cursor-not-allowed dark:placeholder:text-zinc-500"
        />
        <div className="flex items-center justify-between px-2 pb-2 pt-1">
          <div className="flex items-center gap-1">
            {onAttach && (
              <>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".pdf,.txt,.md"
                  multiple
                  className="hidden"
                  onChange={(e) => {
                    onAttach(e.target.files);
                    if (fileInputRef.current) fileInputRef.current.value = '';
                  }}
                />
                <button
                  type="button"
                  onClick={() => fileInputRef.current?.click()}
                  className="inline-flex h-8 w-8 items-center justify-center rounded-lg text-zinc-500 hover:bg-zinc-100 hover:text-zinc-800 dark:hover:bg-zinc-800 dark:hover:text-zinc-200"
                  aria-label="Attach files"
                >
                  <Paperclip size={16} />
                </button>
              </>
            )}
          </div>
          <motion.button
            type="button"
            onClick={submit}
            disabled={!canSend}
            whileHover={{ scale: canSend ? 1.02 : 1 }}
            whileTap={{ scale: canSend ? 0.97 : 1 }}
            className={cn(
              'inline-flex h-9 items-center justify-center gap-1 rounded-xl px-3 text-sm font-medium transition',
              canSend
                ? 'bg-amber-500 text-zinc-950 hover:bg-amber-400'
                : 'bg-zinc-100 text-zinc-400 dark:bg-zinc-800 dark:text-zinc-500',
            )}
            aria-label="Send message"
          >
            {isLoading ? <Loader size={14} className="animate-spin" /> : <Send size={14} />}
            <span className="hidden sm:inline">{isLoading ? 'Thinking' : 'Send'}</span>
          </motion.button>
        </div>
      </div>
      <div className="px-2 pt-1.5 text-[11px] text-zinc-500 dark:text-zinc-500">
        Press <kbd className="rounded border border-zinc-200 bg-zinc-50 px-1 text-[10px] dark:border-zinc-700 dark:bg-zinc-800">Enter</kbd> to send ·{' '}
        <kbd className="rounded border border-zinc-200 bg-zinc-50 px-1 text-[10px] dark:border-zinc-700 dark:bg-zinc-800">Shift+Enter</kbd> for newline
      </div>
    </div>
  );
}
