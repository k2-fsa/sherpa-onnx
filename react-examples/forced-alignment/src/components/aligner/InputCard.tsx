import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Textarea } from "@/components/ui/textarea";
import { Mic, Square, Upload } from "lucide-react";
import { useCallback } from "react";
import { useDropzone } from "react-dropzone";

interface Props {
  fileName: string | null;
  transcript: string;
  onTranscript: (t: string) => void;
  onPickFile: (file: File) => void;
  onExample: () => void;
  onRecordToggle: () => void;
  recording: boolean;
  recordSeconds: number;
  onAlign: () => void;
  onCancel: () => void;
  onClear: () => void;
  isAligning: boolean;
  ready: boolean;
}

export function InputCard(props: Props) {
  const {
    fileName,
    transcript,
    onTranscript,
    onPickFile,
    onExample,
    onRecordToggle,
    recording,
    recordSeconds,
    onAlign,
    onCancel,
    onClear,
    isAligning,
    ready,
  } = props;

  const onDrop = useCallback(
    (files: File[]) => {
      if (files.length > 0) onPickFile(files[0]);
    },
    [onPickFile],
  );
  const dropzone = useDropzone({ onDrop, noClick: true });

  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-base">Input</CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="flex gap-2 flex-wrap">
          <Button variant="outline" onClick={onExample} disabled={!ready}>
            Example
          </Button>
          <Button
            variant={recording ? "destructive" : "outline"}
            onClick={onRecordToggle}
            disabled={!ready}
          >
            {recording ? (
              <>
                <Square className="w-4 h-4 mr-1" /> Stop recording ({recordSeconds}s)
              </>
            ) : (
              <>
                <Mic className="w-4 h-4 mr-1" /> Record from microphone
              </>
            )}
          </Button>
        </div>

        <div
          {...dropzone.getRootProps({
            "aria-label":
              "Audio or video file: drop a file here or use the select button",
          })}
          className={`border border-dashed rounded-lg p-6 text-center cursor-pointer ${
            dropzone.isDragActive ? "border-indigo-500 bg-indigo-50" : "border-gray-300"
          }`}
        >
          <input {...dropzone.getInputProps()} />
          <Upload className="w-5 h-5 mx-auto text-muted-foreground" />
          <p className="mt-2 text-sm text-muted-foreground">
            Drag an audio/video file here, or
            <button
              type="button"
              className="text-indigo-600 underline mx-1 cursor-pointer bg-transparent border-0 p-0 font-normal"
              onClick={(e) => {
                e.stopPropagation();
                dropzone.open();
              }}
            >
              select a file
            </button>
            （wav / mp3 / m4a / flac / ogg / mp4 …）
          </p>
          {fileName && (
            <p className="mt-1 text-sm font-medium truncate">{fileName}</p>
          )}
        </div>

        <Textarea
          value={transcript}
          onChange={(e) => onTranscript(e.target.value)}
          placeholder="Enter the transcript of the audio (Chinese is tokenized per hanzi, English words are split on whitespace; punctuation is ignored)"
          rows={3}
        />

        <div className="flex gap-2 flex-wrap">
          <Button
            onClick={isAligning ? onCancel : onAlign}
            disabled={!ready}
          >
            {isAligning ? "Stop" : "Align"}
          </Button>
          <Button variant="outline" onClick={onClear}>
            Clear
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}
