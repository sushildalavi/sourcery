-- 002_workspace_isolation.sql
--
-- Adds an opt-in workspace_id column on every row that holds tenant data.
-- Existing rows get workspace_id = 'default' so the upgrade is non-breaking.
-- Routes that don't supply X-Workspace-Id continue to read/write the default
-- workspace exactly as before.
--
-- Per-tenant calibration: confidence_calibration also gets workspace_id so
-- different research groups can fit MSA weights against their own corpus
-- and have those weights applied automatically at scoring time.

-- ── core data ─────────────────────────────────────────────────────────────
ALTER TABLE documents
  ADD COLUMN IF NOT EXISTS workspace_id TEXT NOT NULL DEFAULT 'default';

CREATE INDEX IF NOT EXISTS idx_documents_workspace
  ON documents(workspace_id);

ALTER TABLE chunks
  ADD COLUMN IF NOT EXISTS workspace_id TEXT NOT NULL DEFAULT 'default';

CREATE INDEX IF NOT EXISTS idx_chunks_workspace
  ON chunks(workspace_id);

-- ── ancillary state that MUST also be tenant-scoped ──────────────────────
ALTER TABLE chat_sessions
  ADD COLUMN IF NOT EXISTS workspace_id TEXT NOT NULL DEFAULT 'default';

CREATE INDEX IF NOT EXISTS idx_chat_sessions_workspace
  ON chat_sessions(workspace_id);

ALTER TABLE user_memory
  ADD COLUMN IF NOT EXISTS workspace_id TEXT NOT NULL DEFAULT 'default';

CREATE INDEX IF NOT EXISTS idx_user_memory_workspace
  ON user_memory(workspace_id);

ALTER TABLE digests
  ADD COLUMN IF NOT EXISTS workspace_id TEXT NOT NULL DEFAULT 'default';

CREATE INDEX IF NOT EXISTS idx_digests_workspace
  ON digests(workspace_id);

-- ── per-tenant calibration weights ───────────────────────────────────────
ALTER TABLE confidence_calibration
  ADD COLUMN IF NOT EXISTS workspace_id TEXT NOT NULL DEFAULT 'default';

CREATE INDEX IF NOT EXISTS idx_calibration_workspace
  ON confidence_calibration(workspace_id, label, created_at DESC);
