import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import type { WordInterval } from "@/lib/sherpa/engine";
import { useEffect, useRef, useState } from "react";

interface Props {
  /** URL of the picked file (for the preview player) */
  mediaUrl: string | null;
  fileName: string | null;
  words: WordInterval[];
  subtitlesOn: boolean;
  onSubtitlesToggle: () => void;
}

/** Whole-file preview with the current word shown as a subtitle. */
export function MediaPreview({
  mediaUrl,
  fileName,
  words,
  subtitlesOn,
  onSubtitlesToggle,
}: Props) {
  const audioRef = useRef<HTMLAudioElement>(null);
  const [currentWord, setCurrentWord] = useState<string | null>(null);

  useEffect(() => {
    const el = audioRef.current;
    if (!el) return;
    const onTime = () => {
      const t = el.currentTime;
      const w = words.find((w) => t >= w.start && t < w.end);
      setCurrentWord(w ? w.text : null);
    };
    el.addEventListener("timeupdate", onTime);
    return () => el.removeEventListener("timeupdate", onTime);
  }, [words]);

  if (!mediaUrl) return null;

  return (
    <Card>
      <CardHeader className="pb-2">
        <div className="flex items-center">
          <CardTitle className="text-base">Preview</CardTitle>
          <label className="ml-auto flex items-center gap-1 text-xs text-muted-foreground">
            <input
              type="checkbox"
              checked={subtitlesOn}
              onChange={onSubtitlesToggle}
            />
            Subtitles (current word)
          </label>
        </div>
      </CardHeader>
      <CardContent className="space-y-2">
        <audio ref={audioRef} src={mediaUrl} controls className="w-full" />
        {fileName && (
          <p className="text-xs text-muted-foreground truncate">{fileName}</p>
        )}
        {subtitlesOn && currentWord && (
          <p className="text-center text-lg font-semibold text-indigo-700 bg-indigo-50 rounded py-1">
            {currentWord}
          </p>
        )}
      </CardContent>
    </Card>
  );
}
