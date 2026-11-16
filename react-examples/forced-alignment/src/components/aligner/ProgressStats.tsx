import { Card, CardContent } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";

export interface RunStats {
  audioDuration: number;
  decodeSec: number;
  alignSec: number;
  wordCount: number;
}

interface Props {
  status: string;
  error: string | null;
  isAligning: boolean;
  progress: number;
  stats: RunStats | null;
}

function StatCard({ label, value }: { label: string; value: string }) {
  return (
    <Card className="flex-1 min-w-[120px]">
      <CardContent className="p-3 text-center">
        <p className="text-xs text-muted-foreground">{label}</p>
        <p className="mt-1 text-lg font-semibold">{value}</p>
      </CardContent>
    </Card>
  );
}

export function ProgressStats({ status, error, isAligning, progress, stats }: Props) {
  return (
    <Card>
      <CardContent className="p-4 space-y-3">
        <div className="flex items-center gap-3">
          <p className="text-sm text-muted-foreground flex-1">{status}</p>
        </div>
        {isAligning && <Progress value={progress} className="h-2" />}
        {error && <p className="text-sm text-red-600">{error}</p>}
        {stats && (
          <div className="flex gap-3 flex-wrap">
            <StatCard label="Words" value={String(stats.wordCount)} />
            <StatCard
              label="Audio duration"
              value={`${stats.audioDuration.toFixed(2)} s`}
            />
            <StatCard label="Decode" value={`${stats.decodeSec.toFixed(3)} s`} />
            <StatCard label="Align" value={`${stats.alignSec.toFixed(3)} s`} />
            <StatCard
              label="RTF"
              value={
                stats.audioDuration > 0
                  ? (
                      (stats.decodeSec + stats.alignSec) /
                      stats.audioDuration
                    ).toFixed(3)
                  : "0"
              }
            />
          </div>
        )}
      </CardContent>
    </Card>
  );
}
