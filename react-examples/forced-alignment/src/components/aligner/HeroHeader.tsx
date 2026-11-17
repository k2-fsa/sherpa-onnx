import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import type { LoadProgress } from "@/lib/sherpa/engine";

interface Props {
  load: LoadProgress | null;
  ready: boolean;
  error: string | null;
}

export function HeroHeader({ load, ready, error }: Props) {
  const percent =
    load && load.total > 0 ? Math.round((load.loaded / load.total) * 100) : 0;
  return (
    <section className="pt-10 pb-6 text-center">
      <div className="flex justify-center gap-2 flex-wrap">
        {["sherpa-onnx", "Forced Alignment", "WebAssembly"].map((t) => (
          <Badge key={t} variant="secondary" className="font-normal">
            {t}
          </Badge>
        ))}
      </div>
      <h1 className="mt-4 text-3xl md:text-4xl font-bold tracking-tight">
        Forced Alignment <span className="text-indigo-600">Web Demo</span>
      </h1>
      <p className="mt-3 text-muted-foreground max-w-2xl mx-auto">
        Upload an audio/video file or record from the microphone, enter its
        transcript, and get the start/end time of every word. Inference runs
        locally in your browser; nothing is uploaded to any server.
      </p>
      <div className="mt-3 flex justify-center gap-2 flex-wrap text-xs text-muted-foreground">
        {["On-device inference", "No backend", "Many formats", "Offline capable"].map((t) => (
          <Badge key={t} variant="outline" className="font-normal">
            {t}
          </Badge>
        ))}
      </div>
      <div className="mt-6 max-w-xl mx-auto">
        {error ? (
          <p className="text-sm text-red-600">{error}</p>
        ) : ready ? (
          <p className="text-sm text-green-600">Model loaded — ready to align</p>
        ) : (
          <>
            <Progress value={percent} className="h-2" />
            <p className="mt-2 text-xs text-muted-foreground">
              {load ? load.message : "Loading the wasm engine and model…"}
            </p>
          </>
        )}
      </div>
    </section>
  );
}
