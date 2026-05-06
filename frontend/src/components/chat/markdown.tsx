import type { ReactNode } from 'react';
import type { Citation } from '../../api/types';

interface RenderOpts {
  onCite?: (id: number) => void;
  citations?: Citation[];
}

function shortTitle(raw: string | undefined): string {
  if (!raw) return '';
  // 15_LLMasJudge.pdf → LLMasJudge. Strip common prefixes (NN_) + extensions.
  let s = raw.trim();
  s = s.replace(/\.[a-z0-9]{2,4}$/i, '');
  s = s.replace(/^\d{1,3}[_\-\s]+/, '');
  if (s.length > 22) s = s.slice(0, 20) + '…';
  return s;
}

function citationTooltip(id: number, citations?: Citation[]): string {
  const hit = (citations || []).find((c) => (c.id || 0) === id);
  if (!hit) return `Jump to source ${id}`;
  const short = shortTitle(hit.title) || `Source ${id}`;
  const page = typeof hit.page === 'number' ? ` · p.${hit.page}` : '';
  const pct = hit.msa?.msa_score != null
    ? Math.round(hit.msa.msa_score * 100)
    : hit.confidence_percent != null
      ? Math.round(hit.confidence_percent)
      : null;
  return `${short}${page}${pct != null ? ` · support ${pct}%` : ''}`;
}

function renderInline(text: string, opts?: RenderOpts): ReactNode {
  const nodes: ReactNode[] = [];
  const pattern = /(\*\*(.+?)\*\*|\*(.+?)\*|`([^`]+)`|\[S?(\d+)\])/g;
  let last = 0;
  let m: RegExpExecArray | null;
  while ((m = pattern.exec(text)) !== null) {
    if (m.index > last) nodes.push(text.slice(last, m.index));
    if (m[2] != null) nodes.push(<strong key={m.index}>{m[2]}</strong>);
    else if (m[3] != null) nodes.push(<em key={m.index}>{m[3]}</em>);
    else if (m[4] != null) nodes.push(<code key={m.index}>{m[4]}</code>);
    else if (m[5] != null) {
      const id = Number.parseInt(m[5], 10);
      const tooltip = citationTooltip(id, opts?.citations);
      nodes.push(
        <button
          key={m.index}
          type="button"
          onClick={(e) => {
            e.stopPropagation();
            opts?.onCite?.(id);
          }}
          className="cite-chip"
          title={tooltip}
          aria-label={tooltip}
        >
          {id}
        </button>,
      );
    }
    last = m.index + m[0].length;
  }
  if (last < text.length) nodes.push(text.slice(last));
  return nodes.length === 1 ? nodes[0] : <>{nodes}</>;
}

/**
 * Normalize LLM markdown before line-by-line parsing.
 *
 * The generator sometimes emits headings (`### Title`) or horizontal rules
 * (`---`) inline with surrounding prose — no newline before them — and the
 * line-based parser renders them literally. Likewise, stray `##`/`#` without
 * a following space aren't valid headings but leak through as text.
 */
function normalizeMarkdown(raw: string): string {
  let s = raw || '';
  // 1. Insert a blank line before any heading marker that appears mid-line.
  //    Matches `### Title`, `## Title`, etc. preceded by a non-whitespace /
  //    non-hash char so valid line-start headings and hash runs aren't split.
  s = s.replace(/([^\s\n#])[ \t]*(#{1,6}[ \t]+\S)/g, '$1\n\n$2');
  // 2. Strip trailing closing hashes on heading lines: "## Title ##" → "## Title".
  s = s.replace(/^(#{1,6}[ \t]+.+?)\s+#{1,6}\s*$/gm, '$1');
  // 3. Strip orphan hash runs (e.g. stray `##` at end of line) that are neither
  //    a heading prefix nor part of identifier-like tokens (#!shebang, #define).
  //    The run must be (a) bounded by # on neither side, (b) followed by
  //    whitespace or end of string (excluding identifiers), and (c) not
  //    followed by `[ \t]+\S` (which would make it a valid heading).
  s = s.replace(/(?<!#)(#{1,6})(?!#)(?=\s|$)(?![ \t]+\S)/g, '');
  // 4. Collapse 3+ blank lines to a single blank line.
  s = s.replace(/\n{3,}/g, '\n\n');
  return s;
}

export function renderMarkdown(raw: string, opts?: RenderOpts): ReactNode {
  const lines = normalizeMarkdown(raw || '').split('\n');
  const out: ReactNode[] = [];
  let i = 0;
  while (i < lines.length) {
    const line = lines[i];
    const trimmed = line.trim();
    if (!trimmed) {
      i++;
      continue;
    }

    if (/^[-*_]{3,}$/.test(trimmed)) {
      out.push(<hr key={i} />);
      i++;
      continue;
    }

    const hm = line.match(/^(#{1,6})\s+(.+)/);
    if (hm) {
      const lvl = Math.min(hm[1].length, 3);
      const Tag = `h${lvl}` as 'h1' | 'h2' | 'h3';
      out.push(<Tag key={i}>{renderInline(hm[2], opts)}</Tag>);
      i++;
      continue;
    }

    if (line.startsWith('> ')) {
      out.push(<blockquote key={i}>{renderInline(line.slice(2), opts)}</blockquote>);
      i++;
      continue;
    }

    if (/^[-*+]\s/.test(line)) {
      const items: ReactNode[] = [];
      while (i < lines.length && /^[-*+]\s/.test(lines[i])) {
        items.push(<li key={i}>{renderInline(lines[i].replace(/^[-*+]\s+/, ''), opts)}</li>);
        i++;
      }
      out.push(<ul key={`ul${i}`}>{items}</ul>);
      continue;
    }

    if (/^\d+[.)]\s/.test(line)) {
      const items: ReactNode[] = [];
      while (i < lines.length && /^\d+[.)]\s/.test(lines[i])) {
        items.push(<li key={i}>{renderInline(lines[i].replace(/^\d+[.)]\s+/, ''), opts)}</li>);
        i++;
      }
      out.push(<ol key={`ol${i}`}>{items}</ol>);
      continue;
    }

    const pLines: string[] = [];
    while (i < lines.length) {
      const l = lines[i];
      const t = l.trim();
      if (
        !t ||
        /^[-*_]{3,}$/.test(t) ||
        /^#{1,6}\s/.test(l) ||
        l.startsWith('> ') ||
        /^[-*+]\s/.test(l) ||
        /^\d+[.)]\s/.test(l)
      ) {
        break;
      }
      pLines.push(l);
      i++;
    }
    if (pLines.length) out.push(<p key={`p${i}`}>{renderInline(pLines.join(' '), opts)}</p>);
  }
  return <div className="md">{out}</div>;
}
