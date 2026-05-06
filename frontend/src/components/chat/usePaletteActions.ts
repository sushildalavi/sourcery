import { useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import { BarChart3, FileText, Globe } from 'lucide-react';
import type { PaletteAction } from './CommandPalette';

/**
 * Default palette for the main app — navigate between modes, start a new
 * conversation, open analytics. Caller can extend `extra` to pass
 * route-specific actions (e.g. clear chat, toggle evidence panel).
 */
export function useDefaultPaletteActions(
  extra: PaletteAction[] = [],
): PaletteAction[] {
  const navigate = useNavigate();
  return useMemo<PaletteAction[]>(
    () => [
      {
        id: 'nav-public',
        label: 'Switch to Public research mode',
        description: 'Query across arXiv, Semantic Scholar, OpenAlex, Crossref',
        icon: Globe,
        keywords: ['public', 'mode', 'research', 'arxiv', 's2'],
        run: () => navigate('/public'),
      },
      {
        id: 'nav-uploaded',
        label: 'Switch to My documents mode',
        description: 'Query your uploaded corpus',
        icon: FileText,
        keywords: ['uploaded', 'mode', 'docs', 'my'],
        run: () => navigate('/'),
      },
      {
        id: 'nav-analytics',
        label: 'Open Analytics',
        description: 'Calibration, latency, faithfulness dashboards',
        icon: BarChart3,
        keywords: ['analytics', 'metrics', 'calibration'],
        run: () => navigate('/analytics'),
      },
      ...extra,
    ],
    [navigate, extra],
  );
}
