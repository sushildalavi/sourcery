import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '../ui/Card';
import type { JudgeRunSummary } from '../../api/types';

interface FaithfulnessChartProps {
  runs: JudgeRunSummary[];
}

export function FaithfulnessChart({ runs }: FaithfulnessChartProps) {
  const data = runs
    .slice(0, 10)
    .reverse()
    .map((r) => ({
      name: `#${r.id}`,
      Coverage: Math.round((r.metrics?.mean_coverage || 0) * 100),
      Overall: Math.round((r.metrics?.mean_overall_score || 0) * 100),
      Unsupported: r.metrics?.unsupported_total || 0,
    }));

  return (
    <Card>
      <CardHeader>
        <CardTitle>Judge runs — faithfulness & coverage</CardTitle>
        <CardDescription>Overall and coverage scores across the last {data.length || 0} judge runs.</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="h-64 w-full">
          {data.length === 0 ? (
            <div className="flex h-full items-center justify-center text-sm text-zinc-500 dark:text-zinc-400">
              No judge runs recorded yet.
            </div>
          ) : (
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={data} margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
                <CartesianGrid strokeDasharray="3 3" strokeOpacity={0.15} />
                <XAxis dataKey="name" stroke="currentColor" fontSize={11} tickLine={false} axisLine={false} />
                <YAxis stroke="currentColor" fontSize={11} tickLine={false} axisLine={false} />
                <Tooltip
                  contentStyle={{
                    background: 'rgba(24,24,27,0.95)',
                    border: '1px solid rgba(161,161,170,0.2)',
                    borderRadius: 12,
                    color: '#fafafa',
                    fontSize: 12,
                  }}
                  cursor={{ fill: 'rgba(245,158,11,0.08)' }}
                />
                <Legend wrapperStyle={{ fontSize: 11 }} />
                <Bar dataKey="Overall" fill="#f59e0b" radius={[4, 4, 0, 0]} />
                <Bar dataKey="Coverage" fill="#10b981" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
