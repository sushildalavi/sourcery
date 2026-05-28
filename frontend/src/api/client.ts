import {
  AnswerResponse,
  ChunkResult,
  DocumentRow,
  EvalCase,
  EvalRunResponse,
  JudgeRunPayload,
  JudgeRunResponse,
  JudgeRunSummary,
  MsaCalibrationPayload,
  MsaCalibrationResponse,
  MsaCalibrationLatest,
} from './types';

export const API_BASE =
  import.meta.env.VITE_API_BASE_URL ||
  import.meta.env.VITE_API_BASE ||
  'http://127.0.0.1:8000';

const WORKSPACE_STORAGE_KEY = 'sourcery-workspace-id';

/**
 * Resolve the active workspace id for outbound requests, in this order:
 *   1. VITE_WORKSPACE_ID (build-time pin — useful for single-tenant deploys)
 *   2. localStorage[sourcery-workspace-id] (set via setWorkspaceId())
 * Falls back to undefined, in which case the backend uses "default".
 */
export function getWorkspaceId(): string | undefined {
  const fromEnv = import.meta.env.VITE_WORKSPACE_ID as string | undefined;
  if (fromEnv) return fromEnv;
  if (typeof window === 'undefined') return undefined;
  try {
    return window.localStorage.getItem(WORKSPACE_STORAGE_KEY) || undefined;
  } catch {
    return undefined;
  }
}

export function setWorkspaceId(id: string | null): void {
  if (typeof window === 'undefined') return;
  try {
    if (id) window.localStorage.setItem(WORKSPACE_STORAGE_KEY, id);
    else window.localStorage.removeItem(WORKSPACE_STORAGE_KEY);
  } catch {
    /* localStorage disabled — silently no-op */
  }
}

function buildHeaders(extra: HeadersInit | undefined, includeJson: boolean): Record<string, string> {
  const headers: Record<string, string> = {};
  if (includeJson) headers['Content-Type'] = 'application/json';
  const ws = getWorkspaceId();
  if (ws) headers['X-Workspace-Id'] = ws;
  if (extra) {
    for (const [k, v] of Object.entries(extra as Record<string, string>)) {
      headers[k] = v;
    }
  }
  return headers;
}

async function jsonRequest<T>(path: string, opts: RequestInit = {}): Promise<T> {
  try {
    const res = await fetch(`${API_BASE}${path}`, {
      ...opts,
      headers: buildHeaders(opts.headers, true),
    });
    if (!res.ok) {
      throw new Error((await res.text()) || res.statusText);
    }
    return res.json() as Promise<T>;
  } catch (e: any) {
    const msg = e?.message || 'Network error';
    if (/Failed to fetch/i.test(msg)) {
      throw new Error(`Backend unreachable at ${API_BASE}`);
    }
    throw new Error(msg);
  }
}

export const api = {
  async listDocs(limit = 12): Promise<{ documents: DocumentRow[] }> {
    return jsonRequest(`/documents/latest?limit=${limit}`);
  },

  async uploadFile(file: File): Promise<{ document_id: number; pages: number; chunks: number }> {
    const form = new FormData();
    form.append('file', file);
    // Multipart: don't set Content-Type (the browser writes the boundary).
    // We still want the workspace header to flow.
    const res = await fetch(`${API_BASE}/documents/upload`, {
      method: 'POST',
      body: form,
      headers: buildHeaders(undefined, false),
    });
    if (!res.ok) throw new Error(await res.text());
    return res.json();
  },

  async searchChunks(q: string, k = 10, docId?: number): Promise<{ results: ChunkResult[] }> {
    return jsonRequest('/documents/search/chunks', {
      method: 'POST',
      body: JSON.stringify({ q, k, doc_id: docId }),
    });
  },

  async askAssistant(payload: {
    query: string;
    scope: 'uploaded' | 'public';
    doc_id?: number;
    doc_ids?: number[];
    k?: number;
    sense?: string;
    compare_senses?: boolean;
    allow_general_background?: boolean;
    run_judge?: boolean;
    run_judge_llm?: boolean;
  }): Promise<AnswerResponse> {
    return jsonRequest('/assistant/answer', {
      method: 'POST',
      body: JSON.stringify(payload),
    });
  },

  async deleteDoc(docId: number): Promise<{ ok: boolean }> {
    // Must include the workspace header — otherwise the backend rejects
    // the delete (or, worse, falls back to "default" and deletes a
    // doc the caller doesn't own).
    const res = await fetch(`${API_BASE}/documents/${docId}`, {
      method: 'DELETE',
      headers: buildHeaders(undefined, false),
    });
    if (!res.ok) throw new Error(await res.text());
    return res.json();
  },

  async runEval(payload: { name?: string; scope?: 'uploaded' | 'public'; k?: number; cases: EvalCase[] }): Promise<EvalRunResponse> {
    return jsonRequest('/eval/run', {
      method: 'POST',
      body: JSON.stringify(payload),
    });
  },

  async listEvalRuns(limit = 20): Promise<{ runs: EvalRunResponse[] }> {
    return jsonRequest(`/eval/runs?limit=${limit}`);
  },

  async runJudge(payload: JudgeRunPayload): Promise<JudgeRunResponse> {
    return jsonRequest('/eval/judge', {
      method: 'POST',
      body: JSON.stringify(payload),
    });
  },

  async listJudgeRuns(limit = 20): Promise<{ runs: JudgeRunSummary[] }> {
    return jsonRequest(`/eval/judge/runs?limit=${limit}`);
  },

  async calibrateConfidence(payload: MsaCalibrationPayload): Promise<MsaCalibrationResponse> {
    return jsonRequest('/confidence/calibrate', {
      method: 'POST',
      body: JSON.stringify(payload),
    });
  },

  async getLatestCalibration(label?: string): Promise<MsaCalibrationLatest> {
    const qs = label ? `?label=${encodeURIComponent(label)}` : '';
    return jsonRequest(`/confidence/calibration${qs}`);
  },
};
