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

interface RecallChartProps {
  recallOnly: Record<string, number>;
  recallRerank: Record<string, number>;
}

export function RecallChart({ recallOnly, recallRerank }: RecallChartProps) {
  const ks = ['1', '3', '5', '10'];
  const data = ks.map((k) => ({
    k: `@${k}`,
    Retrieval: Number((recallOnly?.[k] ?? 0).toFixed(3)),
    'With Rerank': Number((recallRerank?.[k] ?? 0).toFixed(3)),
  }));

  return (
    <Card>
      <CardHeader>
        <CardTitle>Recall @ K</CardTitle>
        <CardDescription>Retrieval-only vs. reranker on the pinned test set.</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="h-64 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={data} margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" strokeOpacity={0.15} />
              <XAxis dataKey="k" stroke="currentColor" fontSize={11} tickLine={false} axisLine={false} />
              <YAxis stroke="currentColor" fontSize={11} tickLine={false} axisLine={false} domain={[0, 1]} />
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
              <Bar dataKey="Retrieval" fill="#a1a1aa" radius={[6, 6, 0, 0]} />
              <Bar dataKey="With Rerank" fill="#f59e0b" radius={[6, 6, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </CardContent>
    </Card>
  );
}
