import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { formatClock, wordsToTimestampedTxt } from "@/lib/sherpa/formats";
import type { AlignResult, WordInterval } from "@/lib/sherpa/engine";
import { Download, Play, Save, Share2, Square } from "lucide-react";
import type React from "react";

export type ViewFormat = "table" | "timestamped" | "plain";

interface Props {
  result: AlignResult | null;
  playingIndex: number;
  playingAll: boolean;
  onPlayingAll: (v: boolean) => void;
  onPlayWord: (index: number) => void;
  onSaveWord: (index: number) => void;
  onRowClick: (index: number) => void;
  viewFormat: ViewFormat;
  onViewFormat: (f: ViewFormat) => void;
  search: string;
  onSearch: (s: string) => void;
  hitIndex: number;
  onHitIndex: (i: number) => void;
  onExport: (kind: "textgrid" | "timestamped" | "plain" | "textgrid-txt" | "words-txt") => void;
  onShare: (kind: "textgrid-txt" | "words-txt") => void;
  onClearResults: () => void;
  logs: string[];
}

export function ResultPanel(props: Props) {
  const {
    result,
    playingIndex,
    playingAll,
    onPlayingAll,
    onPlayWord,
    onSaveWord,
    onRowClick,
    viewFormat,
    onViewFormat,
    search,
    onSearch,
    hitIndex,
    onHitIndex,
    onExport,
    onShare,
    onClearResults,
    logs,
  } = props;

  const words: WordInterval[] = result ? result.words : [];
  const query = search.trim();
  const hits = query
    ? words
        .map((w, i) => (w.text.toLowerCase().includes(query.toLowerCase()) ? i : -1))
        .filter((i) => i >= 0)
    : [];

  const highlight = (text: string) => {
    if (!query) return text;
    const idx = text.toLowerCase().indexOf(query.toLowerCase());
    if (idx < 0) return text;
    return (
      <>
        {text.slice(0, idx)}
        <mark className="bg-yellow-200">{text.slice(idx, idx + query.length)}</mark>
        {text.slice(idx + query.length)}
      </>
    );
  };

  return (
    <div className="space-y-3">
      <Card>
        <CardHeader className="pb-2">
          <div className="flex items-center gap-2 flex-wrap">
            <CardTitle className="text-base">Result</CardTitle>
            <Tabs
              value={viewFormat}
              onValueChange={(v) => onViewFormat(v as ViewFormat)}
            >
              <TabsList className="h-8">
                <TabsTrigger value="table" className="text-xs">
                  Table
                </TabsTrigger>
                <TabsTrigger value="timestamped" className="text-xs">
                  Timestamped
                </TabsTrigger>
                <TabsTrigger value="plain" className="text-xs">
                  Plain
                </TabsTrigger>
              </TabsList>
            </Tabs>
            <div className="flex items-center gap-1 ml-auto">
              <Input
                value={search}
                onChange={(e: React.ChangeEvent<HTMLInputElement>) => {
                  onSearch(e.target.value);
                  onHitIndex(-1);
                }}
                placeholder="Search…"
                className="h-8 w-32 text-xs"
              />
              <span className="text-xs text-muted-foreground w-10">
                {hits.length > 0 ? `${hitIndex + 1}/${hits.length}` : ""}
              </span>
              <Button
                variant="ghost"
                size="sm"
                disabled={hits.length === 0}
                onClick={() =>
                  onHitIndex((hitIndex - 1 + hits.length) % hits.length)
                }
              >
                ↑
              </Button>
              <Button
                variant="ghost"
                size="sm"
                disabled={hits.length === 0}
                onClick={() => onHitIndex((hitIndex + 1) % hits.length)}
              >
                ↓
              </Button>
            </div>
          </div>
        </CardHeader>
        <CardContent className="space-y-2">
          {!result ? (
            <p className="text-sm text-muted-foreground">No result yet.</p>
          ) : (
            <>
              <p className="text-xs text-muted-foreground">
                Click a row to play from that word; the Audio column plays just
                that word and stops at its end; the save icon stores the word's
                wav. Word boundaries come from the alignment and are
                sample-accurate.
              </p>
              {viewFormat === "table" && (
                <div>
                  <div className="flex text-xs font-semibold text-muted-foreground border-b pb-1">
                    <span className="w-2/5">Word</span>
                    <span className="w-2/5">Time</span>
                    <span className="w-1/5">Audio</span>
                  </div>
                  {words.map((w, i) => {
                    const playing = playingIndex === i;
                    const hit = hits[hitIndex] === i;
                    return (
                      <div
                        key={i}
                        className={`flex items-center text-sm py-1 border-b cursor-pointer ${
                          hit
                            ? "bg-yellow-100"
                            : playing
                              ? "bg-indigo-50"
                              : "hover:bg-gray-50"
                        }`}
                        onClick={() => onRowClick(i)}
                      >
                        <span className="w-2/5 truncate">{highlight(w.text)}</span>
                        <span className="w-2/5 text-xs text-muted-foreground">
                          [{formatClock(w.start)} → {formatClock(w.end)}]
                        </span>
                        <span className="w-1/5 flex gap-1">
                          <Button
                            variant="ghost"
                            size="sm"
                            className="h-7 px-2"
                            onClick={(e) => {
                              e.stopPropagation();
                              onPlayWord(i);
                            }}
                          >
                            {playing ? (
                              <Square className="w-3.5 h-3.5" />
                            ) : (
                              <Play className="w-3.5 h-3.5" />
                            )}
                          </Button>
                          <Button
                            variant="ghost"
                            size="sm"
                            className="h-7 px-2"
                            onClick={(e) => {
                              e.stopPropagation();
                              onSaveWord(i);
                            }}
                          >
                            <Save className="w-3.5 h-3.5" />
                          </Button>
                        </span>
                      </div>
                    );
                  })}
                </div>
              )}
              {viewFormat === "timestamped" && (
                <pre className="text-xs whitespace-pre-wrap">
                  {wordsToTimestampedTxt(words)}
                </pre>
              )}
              {viewFormat === "plain" && (
                <p className="text-sm leading-7">
                  {words.map((w, i) => (
                    <span
                      key={i}
                      className={`cursor-pointer ${
                        playingIndex === i ? "text-indigo-600 font-semibold" : ""
                      }`}
                      onClick={() => onRowClick(i)}
                    >
                      {highlight(w.text)}{" "}
                    </span>
                  ))}
                </p>
              )}

              <div className="flex gap-2 flex-wrap items-center pt-2">
                <Button variant="outline" size="sm" onClick={() => onExport("textgrid")}>
                  <Download className="w-3.5 h-3.5 mr-1" /> .TextGrid
                </Button>
                <Button variant="outline" size="sm" onClick={() => onExport("timestamped")}>
                  <Download className="w-3.5 h-3.5 mr-1" /> Timestamped .txt
                </Button>
                <Button variant="outline" size="sm" onClick={() => onExport("plain")}>
                  <Download className="w-3.5 h-3.5 mr-1" /> Plain .txt
                </Button>
                <Button variant="outline" size="sm" onClick={() => onShare("textgrid-txt")}>
                  <Share2 className="w-3.5 h-3.5 mr-1" /> Share .TextGrid.txt
                </Button>
                <Button variant="outline" size="sm" onClick={() => onShare("words-txt")}>
                  <Share2 className="w-3.5 h-3.5 mr-1" /> Share .words.txt
                </Button>
                <Button variant="ghost" size="sm" onClick={() => onExport("textgrid-txt")}>
                  Download .TextGrid.txt
                </Button>
                <Button variant="ghost" size="sm" onClick={() => onExport("words-txt")}>
                  Download .words.txt
                </Button>
              </div>
              <div className="flex gap-2 items-center">
                <label className="flex items-center gap-1 text-xs">
                  <input
                    type="checkbox"
                    checked={playingAll}
                    onChange={(e) => onPlayingAll(e.target.checked)}
                  />
                  Play all words in order
                </label>
                <Button variant="ghost" size="sm" className="text-red-600" onClick={onClearResults}>
                  Clear results
                </Button>
              </div>
            </>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-base">Log</CardTitle>
        </CardHeader>
        <CardContent className="space-y-1 max-h-48 overflow-y-auto">
          {logs.length === 0 ? (
            <p className="text-xs text-muted-foreground">No log yet.</p>
          ) : (
            logs.map((line, i) => (
              <p key={i} className="text-xs font-mono text-muted-foreground">
                {line}
              </p>
            ))
          )}
        </CardContent>
      </Card>
    </div>
  );
}
