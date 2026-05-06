import { useRef, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { FileText, Loader, Trash2, Upload } from 'lucide-react';
import type { DocumentRow } from '../../api/types';
import { Dialog } from '../ui/Dialog';
import { Button } from '../ui/Button';
import { cn } from '../../lib/cn';

interface DocumentsPanelProps {
  docs: DocumentRow[];
  selectedDocs: number[];
  onToggle: (id: number) => void;
  onUpload: (files: FileList | File[]) => Promise<void> | void;
  onDelete: (id: number) => Promise<void> | void;
}

export function DocumentsPanel({ docs, selectedDocs, onToggle, onUpload, onDelete }: DocumentsPanelProps) {
  const [dragActive, setDragActive] = useState(false);
  const [pendingDelete, setPendingDelete] = useState<DocumentRow | null>(null);
  const [uploading, setUploading] = useState(false);
  const fileRef = useRef<HTMLInputElement | null>(null);

  const handleFiles = async (files: FileList | File[] | null) => {
    if (!files) return;
    setUploading(true);
    try {
      await onUpload(files);
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="flex h-full w-full flex-col gap-3 p-3">
      <div className="px-1 text-[10px] font-semibold uppercase tracking-wider text-zinc-500 dark:text-zinc-500">
        Documents
      </div>

      <motion.label
        onDragOver={(e) => {
          e.preventDefault();
          setDragActive(true);
        }}
        onDragLeave={() => setDragActive(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragActive(false);
          handleFiles(e.dataTransfer.files);
        }}
        whileHover={{ scale: 1.005 }}
        className={cn(
          'flex cursor-pointer flex-col items-center justify-center gap-1 rounded-2xl border border-dashed px-4 py-5 text-center transition',
          dragActive
            ? 'border-amber-500 bg-amber-500/10'
            : 'border-zinc-300 bg-white hover:border-amber-500/50 dark:border-zinc-700 dark:bg-zinc-900/60',
        )}
      >
        <input
          ref={fileRef}
          type="file"
          accept=".pdf,.txt,.md"
          multiple
          className="hidden"
          onChange={(e) => {
            handleFiles(e.target.files);
            if (fileRef.current) fileRef.current.value = '';
          }}
        />
        {uploading ? (
          <Loader size={18} className="animate-spin text-amber-600 dark:text-amber-400" />
        ) : (
          <Upload size={18} className="text-zinc-500 dark:text-zinc-400" />
        )}
        <div className="text-xs font-medium text-zinc-700 dark:text-zinc-200">
          {uploading ? 'Uploading...' : 'Drop files or click to upload'}
        </div>
        <div className="text-[10px] text-zinc-500 dark:text-zinc-500">PDF · TXT · MD</div>
      </motion.label>

      {docs.length === 0 ? (
        <div className="flex flex-1 items-center justify-center text-center text-[11px] text-zinc-500 dark:text-zinc-500">
          No documents yet. Upload a PDF to get started.
        </div>
      ) : (
        <div className="flex-1 space-y-1.5 overflow-y-auto pr-1">
          <AnimatePresence initial={false}>
            {docs.map((d) => {
              const selected = selectedDocs.includes(d.id);
              const ready = d.status === 'ready';
              const isError = d.status === 'error';
              return (
                <motion.div
                  key={d.id}
                  initial={{ opacity: 0, y: 6 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -6 }}
                  transition={{ duration: 0.2 }}
                  onClick={() => ready && onToggle(d.id)}
                  className={cn(
                    'group flex cursor-pointer items-start gap-2 rounded-xl border px-2.5 py-2 transition',
                    selected
                      ? 'border-amber-500/60 bg-amber-500/10'
                      : 'border-zinc-200 bg-white hover:border-amber-500/30 dark:border-zinc-800 dark:bg-zinc-900/60',
                    !ready && 'opacity-70 cursor-not-allowed',
                  )}
                >
                  <span
                    aria-hidden
                    className={cn(
                      'mt-0.5 flex h-4 w-4 shrink-0 items-center justify-center rounded-md border',
                      selected
                        ? 'border-amber-500 bg-amber-500 text-zinc-950'
                        : 'border-zinc-300 dark:border-zinc-700',
                    )}
                  >
                    {selected && (
                      <svg viewBox="0 0 24 24" width="10" height="10" fill="none" stroke="currentColor" strokeWidth="3">
                        <polyline points="4 12 10 18 20 6" />
                      </svg>
                    )}
                  </span>
                  <div className="min-w-0 flex-1">
                    <div className="flex items-center gap-1.5">
                      <FileText size={11} className="shrink-0 text-zinc-500" />
                      <span className="truncate text-xs font-medium text-zinc-800 dark:text-zinc-100" title={d.title}>
                        {d.title}
                      </span>
                    </div>
                    <div
                      className={cn(
                        'mt-0.5 text-[10px]',
                        ready
                          ? 'text-emerald-600 dark:text-emerald-400'
                          : isError
                            ? 'text-red-500'
                            : 'text-zinc-500',
                      )}
                    >
                      {ready ? 'Ready' : isError ? 'Error' : 'Processing...'}
                    </div>
                  </div>
                  <button
                    type="button"
                    className="opacity-0 transition group-hover:opacity-100 text-zinc-400 hover:text-red-500"
                    onClick={(e) => {
                      e.stopPropagation();
                      setPendingDelete(d);
                    }}
                    aria-label="Delete"
                  >
                    <Trash2 size={13} />
                  </button>
                </motion.div>
              );
            })}
          </AnimatePresence>
        </div>
      )}

      <Dialog
        open={Boolean(pendingDelete)}
        onClose={() => setPendingDelete(null)}
        title="Delete document"
        description={pendingDelete ? `Remove ${pendingDelete.title}?` : undefined}
        footer={
          <>
            <Button variant="ghost" size="sm" onClick={() => setPendingDelete(null)}>
              Cancel
            </Button>
            <Button
              variant="danger"
              size="sm"
              onClick={async () => {
                if (pendingDelete) await onDelete(pendingDelete.id);
                setPendingDelete(null);
              }}
            >
              Delete
            </Button>
          </>
        }
      >
        <p>All chunks and embeddings will be removed permanently.</p>
      </Dialog>
    </div>
  );
}
