// Audio decoding for forced alignment: the browser's native decoder
// handles any audio/video container, then an OfflineAudioContext renders
// 16 kHz mono (what the aligner wants).

export const TARGET_SAMPLE_RATE = 16000;

export interface DecodedAudio {
  /** float32 mono samples in [-1, 1] at 16 kHz */
  samples: Float32Array;
  sampleRate: number;
  duration: number;
}

/**
 * Decodes an audio/video file (wav/mp3/m4a/ogg/… and the audio track of
 * videos) to 16 kHz mono float32 PCM. Returns null when the browser cannot
 * decode the file.
 */
export async function decodeToMono16k(file: File | Blob): Promise<DecodedAudio | null> {
  try {
    const arrayBuffer = await file.arrayBuffer();
    const decodeCtx = new AudioContext();
    let audioBuffer: AudioBuffer;
    try {
      audioBuffer = await decodeCtx.decodeAudioData(arrayBuffer.slice(0));
    } finally {
      decodeCtx.close();
    }

    const frameCount = Math.max(
      1,
      Math.ceil(audioBuffer.duration * TARGET_SAMPLE_RATE),
    );
    const offlineCtx = new OfflineAudioContext(1, frameCount, TARGET_SAMPLE_RATE);
    const src = offlineCtx.createBufferSource();
    src.buffer = audioBuffer;
    src.connect(offlineCtx.destination);
    src.start();
    const rendered = await offlineCtx.startRendering();
    return {
      samples: rendered.getChannelData(0),
      sampleRate: TARGET_SAMPLE_RATE,
      duration: rendered.duration,
    };
  } catch (e) {
    console.warn("decode error", e);
    return null;
  }
}

/** Encodes 16-bit PCM wav bytes (for saving word segments). */
export function encodeWav(samples: Float32Array, sampleRate: number): Uint8Array {
  const dataSize = samples.length * 2;
  const out = new ArrayBuffer(44 + dataSize);
  const view = new DataView(out);
  const writeStr = (offset: number, s: string) => {
    for (let i = 0; i < s.length; ++i) view.setUint8(offset + i, s.charCodeAt(i));
  };
  writeStr(0, "RIFF");
  view.setUint32(4, 36 + dataSize, true);
  writeStr(8, "WAVE");
  writeStr(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true); // PCM
  view.setUint16(22, 1, true); // mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeStr(36, "data");
  view.setUint32(40, dataSize, true);
  let offset = 44;
  for (let i = 0; i < samples.length; ++i, offset += 2) {
    const v = Math.max(-1, Math.min(1, samples[i]));
    // round to the nearest 16-bit step (truncation biases toward zero)
    view.setInt16(offset, Math.round(v < 0 ? v * 32768 : v * 32767), true);
  }
  return new Uint8Array(out);
}

/** Sample-accurate playback of samples[startSec, endSec) via Web Audio. */
export function playSlice(
  samples: Float32Array,
  sampleRate: number,
  startSec: number,
  endSec: number,
  onEnded?: () => void,
): AudioBufferSourceNode | null {
  const s = Math.max(0, Math.floor(startSec * sampleRate));
  const e = Math.min(samples.length, Math.ceil(endSec * sampleRate));
  if (e <= s) {
    onEnded?.();
    return null;
  }
  const slice = samples.slice(s, e);
  const ctx = new AudioContext({ sampleRate });
  const buf = ctx.createBuffer(1, slice.length, sampleRate);
  buf.getChannelData(0).set(slice);
  const src = ctx.createBufferSource();
  src.buffer = buf;
  src.connect(ctx.destination);
  src.onended = () => {
    ctx.close();
    onEnded?.();
  };
  src.start();
  return src;
}
