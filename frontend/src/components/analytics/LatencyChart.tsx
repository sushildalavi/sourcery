import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '../ui/Card';

interface LatencyChartProps {
  breakdown: {
    retrieve_ms_avg: number;
    rerank_ms_avg: number;
    generate_ms_avg: number;
  };
}

export function LatencyChart({ breakdown }: LatencyChartProps) {
  const data = [
    { op: 'Retrieve', ms: Math.round(breakdown?.retrieve_ms_avg || 0) },
    { op: 'Rerank', ms: Math.round(breakdown?.rerank_ms_avg || 0) },
    { op: 'Generate', ms: Math.round(breakdown?.generate_ms_avg || 0) },
  ];

  return (
    <Card>
      <CardHeader>
        <CardTitle>Latency breakdown</CardTitle>
        <CardDescription>Average milliseconds per stage from the latest run.</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="h-64 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={data} layout="vertical" margin={{ top: 8, right: 16, bottom: 8, left: 32 }}>
              <CartesianGrid strokeDasharray="3 3" strokeOpacity={0.15} />
              <XAxis type="number" stroke="currentColor" fontSize={11} tickLine={false} axisLine={false} />
              <YAxis type="category" dataKey="op" stroke="currentColor" fontSize={11} tickLine={false} axisLine={false} />
              <Tooltip
                contentStyle={{
                  background: 'rgba(24,24,27,0.95)',
                  border: '1px solid rgba(161,161,170,0.2)',
                  borderRadius: 12,
                  color: '#fafafa',
                  fontSize: 12,
                }}
                cursor={{ fill: 'rgba(245,158,11,0.08)' }}
                formatter={(value) => [`${value} ms`, 'Latency']}
              />
              <Bar dataKey="ms" fill="#f59e0b" radius={[0, 6, 6, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </CardContent>
    </Card>
  );
}
