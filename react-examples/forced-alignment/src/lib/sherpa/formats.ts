// Result formatting: clock strings and export file contents.

import type { WordInterval } from "./engine";

/** Formats seconds as mm:ss.cc */
export function formatClock(t: number): string {
  // round to centiseconds first so that e.g. 59.996 rolls over to 01:00.00
  // instead of producing the invalid "00:60.00"
  const total = Math.round(t * 100) / 100;
  const m = Math.floor(total / 60);
  const s = total - m * 60;
  return `${String(m).padStart(2, "0")}:${s.toFixed(2).padStart(5, "0")}`;
}

/** One line per word: "word start -- end" (the share format). */
export function wordsToWordsTxt(words: WordInterval[]): string {
  return words
    .map((w) => `${w.text} ${w.start.toFixed(2)} -- ${w.end.toFixed(2)}`)
    .join("\n");
}

/** One line per word: "[mm:ss.cc -> mm:ss.cc] word" */
export function wordsToTimestampedTxt(words: WordInterval[]): string {
  return words
    .map((w) => `[${formatClock(w.start)} -> ${formatClock(w.end)}] ${w.text}`)
    .join("\n");
}

/** The transcript of the words. */
export function wordsToPlainTxt(words: WordInterval[]): string {
  return words.map((w) => w.text).join(" ");
}

/** Triggers a browser download of binary content. */
export function downloadBytes(
  bytes: Uint8Array,
  fileName: string,
  mime = "application/octet-stream",
): void {
  const blob = new Blob([bytes], { type: mime });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = fileName;
  a.click();
  URL.revokeObjectURL(a.href);
}

/** Triggers a browser download of a text file. */
export function downloadText(content: string, fileName: string, mime = "text/plain"): void {
  const blob = new Blob([content], { type: mime });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = fileName;
  a.click();
  URL.revokeObjectURL(a.href);
}

/** Shares a text file with other apps where the Web Share API supports files. */
export async function shareTextFile(
  content: string,
  fileName: string,
): Promise<boolean> {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const nav = navigator as any;
  if (!nav.share || !nav.canShare) return false;
  const file = new File([content], fileName, { type: "text/plain" });
  if (!nav.canShare({ files: [file] })) return false;
  try {
    await nav.share({ files: [file], title: fileName });
    return true;
  } catch {
    return false;
  }
}
