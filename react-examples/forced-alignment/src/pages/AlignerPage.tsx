import { AppFooter } from "@/components/aligner/AppFooter";
import { HeroHeader } from "@/components/aligner/HeroHeader";
import { InputCard } from "@/components/aligner/InputCard";
import { MediaPreview } from "@/components/aligner/MediaPreview";
import {
  type RunStats,
  ProgressStats,
} from "@/components/aligner/ProgressStats";
import {
  type ViewFormat,
  ResultPanel,
} from "@/components/aligner/ResultPanel";
import {
  decodeToMono16k,
  encodeWav,
  playSlice,
  type DecodedAudio,
} from "@/lib/sherpa/audio";
import {
  type AlignResult,
  type LoadProgress,
  type SherpaEngine,
  type VersionInfo,
  readVersionInfo,
  align,
  listModelDirs,
  loadSherpaEngineOnce,
  readExampleWav,
} from "@/lib/sherpa/engine";
import {
  downloadBytes,
  downloadText,
  shareTextFile,
  wordsToPlainTxt,
  wordsToTimestampedTxt,
  wordsToWordsTxt,
} from "@/lib/sherpa/formats";
import { currentModel } from "@/lib/model-config";
import { useEffect, useRef, useState } from "react";

function nowStamp(): string {
  return new Date().toLocaleTimeString("zh-CN", { hour12: false });
}

export default function AlignerPage() {
  // engine
  const [engine, setEngine] = useState<SherpaEngine | null>(null);
  const [load, setLoad] = useState<LoadProgress | null>(null);
  const [engineError, setEngineError] = useState<string | null>(null);
  const [version, setVersion] = useState<VersionInfo | null>(null);

  // input
  const [fileName, setFileName] = useState<string | null>(null);
  const [mediaUrl, setMediaUrl] = useState<string | null>(null);
  const [transcript, setTranscript] = useState("");
  const [decoded, setDecoded] = useState<DecodedAudio | null>(null);
  const [decodeSec, setDecodeSec] = useState(0);

  // recording
  const [recording, setRecording] = useState(false);
  const [recordSeconds, setRecordSeconds] = useState(0);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const recordStartRef = useRef(0);
  const recordTimerRef = useRef<number | null>(null);

  // run
  const [isAligning, setIsAligning] = useState(false);
  const [status, setStatus] = useState("Ready");
  const [error, setError] = useState<string | null>(null);
  const [progress, setProgress] = useState(0);
  const [stats, setStats] = useState<RunStats | null>(null);
  const [logs, setLogs] = useState<string[]>([]);

  // results
  const [result, setResult] = useState<AlignResult | null>(null);
  const [playingIndex, setPlayingIndex] = useState(-1);
  const [playingAll, setPlayingAll] = useState(false);
  const [viewFormat, setViewFormat] = useState<ViewFormat>("table");
  const [search, setSearch] = useState("");
  const [hitIndex, setHitIndex] = useState(-1);
  const [subtitlesOn, setSubtitlesOn] = useState(true);

  const playTokenRef = useRef(0);
  const playingAllRef = useRef(false);
  const playWordRef = useRef<(index: number) => void>(() => {});
  const pickTokenRef = useRef(0);
  const runTokenRef = useRef(0);
  const recordPendingRef = useRef(false);
  const playSourceRef = useRef<AudioBufferSourceNode | null>(null);

  const appendLog = (msg: string) => {
    setLogs((prev) => [...prev, `[${nowStamp()}] ${msg}`].slice(-200));
  };

  // ---- engine loading ----
  useEffect(() => {
    loadSherpaEngineOnce({
      onProgress: setLoad,
      onNativeLog: (line) => appendLog(`[wasm] ${line}`),
    })
      .then((e) => {
        setEngine(e);
        setVersion(readVersionInfo(e.module));
        appendLog(`Model loaded: ${listModelDirs(e.module).join(", ")}`);
      })
      .catch((e) => {
        setEngineError(String(e?.message || e));
      });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ---- input handling ----
  const acceptFile = async (file: File) => {
    const token = ++pickTokenRef.current;
    // a running alignment belongs to the old input: drop it and its busy
    // state, and forget the old samples so they cannot be aligned against
    // the new file name
    runTokenRef.current += 1;
    setIsAligning(false);
    stopPlayback();
    setDecoded(null);
    setResult(null);
    setStats(null);
    setFileName(file.name);
    setMediaUrl((prev: string | null) => {
      if (prev) URL.revokeObjectURL(prev);
      return URL.createObjectURL(file);
    });
    appendLog(`Selected: ${file.name}`);
    setStatus("Decoding…");
    const t0 = performance.now();
    const d = await decodeToMono16k(file);
    const dt = (performance.now() - t0) / 1000;
    if (pickTokenRef.current !== token) return; // superseded by a newer file
    if (!d) {
      setStatus("Decode failed");
      setError("Could not decode this file");
      setDecoded(null);
      appendLog("Decode failed");
      return;
    }
    setDecoded(d);
    setDecodeSec(dt);
    setStatus(`Decoded: ${d.duration.toFixed(2)} s`);
    appendLog(`Decoded: ${d.duration.toFixed(2)} s in ${dt.toFixed(3)} s`);
  };

  const loadExample = async () => {
    if (!engine) return;
    const model = currentModel();
    try {
      const bytes = readExampleWav(engine.module, model.dir, model.exampleWav);
      const file = new File([bytes], model.exampleWav);
      setTranscript(model.exampleText);
      appendLog(`Loaded example: ${model.dir}/${model.exampleWav}`);
      await acceptFile(file);
    } catch (e) {
      setError(`Could not load the example audio: ${e}`);
    }
  };

  const toggleRecord = async () => {
    if (recording) {
      recorderRef.current?.stop();
      return;
    }
    if (recordPendingRef.current) return; // a permission request is in flight
    recordPendingRef.current = true;
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const rec = new MediaRecorder(stream);
      const chunks: Blob[] = [];
      rec.ondataavailable = (e) => {
        if (e.data.size > 0) chunks.push(e.data);
      };
      rec.onstop = async () => {
        stream.getTracks().forEach((t) => t.stop());
        if (recordTimerRef.current) {
          window.clearInterval(recordTimerRef.current);
          recordTimerRef.current = null;
        }
        setRecording(false);
        const blob = new Blob(chunks, { type: rec.mimeType || "audio/webm" });
        appendLog(`Recording finished: ${(blob.size / 1024).toFixed(0)} KB`);
        await acceptFile(new File([blob], "recording.webm"));
      };
      recorderRef.current = rec;
      rec.start();
      recordPendingRef.current = false;
      setRecording(true);
      setRecordSeconds(0);
      recordStartRef.current = Date.now();
      recordTimerRef.current = window.setInterval(() => {
        setRecordSeconds(Math.floor((Date.now() - recordStartRef.current) / 1000));
      }, 500);
      appendLog("Recording started");
    } catch (e) {
      recordPendingRef.current = false;
      setError(`Cannot access the microphone: ${e}`);
    }
  };

  const clearAll = () => {
    runTokenRef.current += 1;
    pickTokenRef.current += 1;
    stopPlayback();
    setIsAligning(false); // the invalidated run's finally will not reset it
    setResult(null);
    setStats(null);
    setFileName(null);
    setMediaUrl((prev: string | null) => {
      if (prev) URL.revokeObjectURL(prev);
      return null;
    });
    setTranscript("");
    setDecoded(null);
    setLogs([]);
    setError(null);
    setStatus("Ready");
    setPlayingIndex(-1);
    recordPendingRef.current = false;
  };

  // ---- alignment ----
  const runAlign = async () => {
    if (!engine) return;
    if (!decoded) {
      const msg = "Please select or record an audio file first";
      setError(msg);
      setStatus(msg);
      appendLog(msg);
      return;
    }
    if (!transcript.trim()) {
      const msg = "Please enter the transcript of the audio";
      setError(msg);
      setStatus(msg);
      appendLog(msg);
      return;
    }
    const token = ++runTokenRef.current;
    playTokenRef.current += 1;
    setError(null);
    setResult(null);
    setPlayingIndex(-1);
    setIsAligning(true);
    setProgress(10);
    setStatus("Aligning…");
    appendLog("Alignment started");
    // yield so the progress bar paints
    await new Promise((r) => setTimeout(r, 30));
    // cancelled while waiting? then do not start the blocking inference
    if (runTokenRef.current !== token) return;
    try {
      const t0 = performance.now();
      const r = align(engine, decoded.samples, decoded.sampleRate, transcript.trim());
      const dt = (performance.now() - t0) / 1000;
      if (runTokenRef.current !== token) return; // cancelled/cleared/re-run
      setProgress(100);
      if (r.words.length === 0) {
        setStatus("Alignment failed");
        setError("Alignment failed: please check that the transcript matches the audio");
        appendLog("Alignment failed (no words)");
      } else {
        setResult(r);
        setStats({
          audioDuration: decoded.duration,
          decodeSec,
          alignSec: dt,
          wordCount: r.words.length,
        });
        setStatus("Alignment complete");
        appendLog(
          `Aligned: ${r.words.length} words | audio ${decoded.duration.toFixed(3)} s | ` +
            `decode ${decodeSec.toFixed(3)} s | align ${dt.toFixed(3)} s | ` +
            `RTF ${((decodeSec + dt) / decoded.duration).toFixed(3)}`,
        );
      }
    } catch (e) {
      if (runTokenRef.current !== token) return;
      setStatus("Alignment failed");
      setError(`Alignment failed: ${e}`);
      appendLog(`Alignment failed: ${e}`);
    } finally {
      if (runTokenRef.current === token) setIsAligning(false);
    }
  };

  // ---- playback (sample-accurate word slices) ----
  const stopPlayback = () => {
    playTokenRef.current += 1;
    const src = playSourceRef.current;
    playSourceRef.current = null;
    if (src) {
      src.onended = null;
      try {
        src.stop();
      } catch {
        // already stopped
      }
    }
    setPlayingIndex(-1);
  };

  const playWord = (index: number) => {
    if (!decoded || !result) return;
    if (playingIndex === index) {
      stopPlayback();
      return;
    }
    const words = result.words;
    if (index < 0 || index >= words.length) return;
    stopPlayback(); // never overlap two words
    const token = playTokenRef.current;
    setPlayingIndex(index);
    const w = words[index];
    playSourceRef.current = playSlice(
      decoded.samples,
      decoded.sampleRate,
      w.start,
      w.end,
      () => {
        playSourceRef.current = null;
        if (playTokenRef.current !== token) return;
        if (playingAllRef.current && index + 1 < words.length) {
          playWordRef.current(index + 1);
        } else {
          setPlayingIndex(-1);
        }
      },
    );
  };

  // the playback chain must always see the latest state
  playWordRef.current = playWord;

  const saveWord = (index: number) => {
    if (!decoded || !result) return;
    const w = result.words[index];
    const s = Math.max(0, Math.floor(w.start * decoded.sampleRate));
    const e = Math.min(decoded.samples.length, Math.ceil(w.end * decoded.sampleRate));
    const bytes = encodeWav(decoded.samples.slice(s, e), decoded.sampleRate);
    const base = (fileName || "audio").replace(/\.[^.]+$/, "");
    downloadBytes(bytes, `${base}_word_${index + 1}.wav`, "audio/wav");
  };

  // ---- exports / share ----
  const baseName = (fileName || "audio").replace(/\.[^.]+$/, "");

  const onExport = (kind: string) => {
    if (!result) return;
    switch (kind) {
      case "textgrid":
        downloadText(result.textgrid, `${baseName}.TextGrid`);
        break;
      case "textgrid-txt":
        downloadText(result.textgrid, `${baseName}.TextGrid.txt`);
        break;
      case "timestamped":
        downloadText(wordsToTimestampedTxt(result.words), `${baseName}_timestamped.txt`);
        break;
      case "plain":
        downloadText(wordsToPlainTxt(result.words), `${baseName}.txt`);
        break;
      case "words-txt":
        downloadText(wordsToWordsTxt(result.words), `${baseName}.words.txt`);
        break;
    }
    appendLog(`Exported: ${kind}`);
  };

  const onShare = async (kind: "textgrid-txt" | "words-txt") => {
    if (!result) return;
    const content =
      kind === "textgrid-txt" ? result.textgrid : wordsToWordsTxt(result.words);
    const name = kind === "textgrid-txt" ? `${baseName}.TextGrid.txt` : `${baseName}.words.txt`;
    const ok = await shareTextFile(content, name);
    if (!ok) {
      // fall back to a download
      downloadText(content, name);
      appendLog(`This browser does not support sharing; downloaded ${name} instead`);
    } else {
      appendLog(`Shared: ${name}`);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-slate-50 to-slate-100">
      <main className="max-w-4xl mx-auto px-4">
        <HeroHeader load={load} ready={!!engine} error={engineError} />
        <div className="space-y-3">
          <InputCard
            fileName={fileName}
            transcript={transcript}
            onTranscript={setTranscript}
            onPickFile={acceptFile}
            onExample={loadExample}
            onRecordToggle={toggleRecord}
            recording={recording}
            recordSeconds={recordSeconds}
            onAlign={runAlign}
            onCancel={() => {
              // the wasm compute() cannot be interrupted; the result is
              // discarded when it arrives (see the token check in runAlign)
              runTokenRef.current += 1;
              setIsAligning(false);
              setStatus("Cancelled");
              appendLog("Cancelled");
            }}
            onClear={clearAll}
            isAligning={isAligning}
            ready={!!engine}
          />
          <ProgressStats
            status={status}
            error={error}
            isAligning={isAligning}
            progress={progress}
            stats={stats}
          />
          <MediaPreview
            mediaUrl={mediaUrl}
            fileName={fileName}
            words={result ? result.words : []}
            subtitlesOn={subtitlesOn}
            onSubtitlesToggle={() => setSubtitlesOn((v: boolean) => !v)}
          />
          <ResultPanel
            result={result}
            playingIndex={playingIndex}
            playingAll={playingAll}
            onPlayingAll={(v: boolean) => {
              playingAllRef.current = v;
              setPlayingAll(v);
            }}
            onPlayWord={playWord}
            onSaveWord={saveWord}
            onRowClick={playWord}
            viewFormat={viewFormat}
            onViewFormat={setViewFormat}
            search={search}
            onSearch={setSearch}
            hitIndex={hitIndex}
            onHitIndex={setHitIndex}
            onExport={onExport}
            onShare={onShare}
            onClearResults={() => {
              runTokenRef.current += 1;
              stopPlayback();
              setIsAligning(false);
              setResult(null);
              setStats(null);
              setPlayingIndex(-1);
            }}
            logs={logs}
          />
          <AppFooter version={version} />
        </div>
      </main>
    </div>
  );
}
