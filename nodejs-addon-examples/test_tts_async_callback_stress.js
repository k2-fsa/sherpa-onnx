// Copyright (c)  2026  Kevin Castillo
//
// Stress test for the async TTS progress-callback path (generateAsync + onProgress).
// Covers: repeated generation, cancellation via callback return value, a throwing
// callback (cancels the generation and rejects the promise with the thrown
// error), concurrent generations, and RSS.
//
// Usage:
//   node test_tts_async_callback_stress.js [model-dir] [iterations]
//
// please download model files from
// https://github.com/k2-fsa/sherpa-onnx/releases/tag/tts-models
//
// The model onnx filename is derived from the directory name (vits-piper-X -> X.onnx),
// so any vits-piper voice works. Short runs (e.g. 30) suit CI; RSS is informational
// unless STRICT_RSS=1 (steady growth with varying shapes is the known ONNX Runtime
// memory-pattern issue #1939, shared with the sync path — not the callback machinery).

const path = require('path');
const sherpa_onnx = require('sherpa-onnx-node');

const modelDir = process.argv[2] || './vits-piper-en_GB-cori-medium';
const iterations = Number(process.argv[3] || 100);

const base = path.basename(path.resolve(modelDir)).replace(/^vits-piper-/, '');
const makeTts = () => new sherpa_onnx.OfflineTts({
  model: {
    vits: {
      model: `${modelDir}/${base}.onnx`,
      tokens: `${modelDir}/tokens.txt`,
      dataDir: `${modelDir}/espeak-ng-data`,
    },
    numThreads: 2,
    provider: 'cpu',
    debug: false,
  },
  maxNumSentences: 1,
});

const TEXT =
    'Hello there, how are you doing on this fine morning today my dear friend?';
const rssMb = () => Math.round(process.memoryUsage().rss / (1024 * 1024));

// (A child_process.fork() harness for the throwing case was removed: in a
// forked child — and only there; spawn() is fine — a pending throwing
// callback coincides with a native crash in a later batch's phonemize,
// reproducible on stock v1.13.4 as well. Pre-existing and fork-specific;
// root cause tracked separately from this fix.)

async function sequential(tts) {
  let totalChunks = 0;
  let steadyStateBase = 0;
  const warmup = Math.min(10, Math.max(1, Math.floor(iterations / 3)));
  for (let i = 0; i < iterations; i++) {
    let chunks = 0;
    // alternate between the generationConfig path and the legacy path so both
    // native workers (TtsGenerateWithConfigWorker / TtsGenerateWorker) run
    const req = i % 2 === 0 ?
        {text: TEXT, generationConfig: {sid: 0, speed: 1.0}} :
        {text: TEXT, sid: 0, speed: 1.0};
    await tts.generateAsync({
      ...req,
      onProgress: () => {
        chunks++;
        return 1;
      },
    });
    totalChunks += chunks;
    if (i === warmup - 1) steadyStateBase = rssMb();
  }
  const growth = rssMb() - steadyStateBase;
  console.log(`sequential: ${iterations} calls, ${
      totalChunks} chunks, steady-state RSS growth ${growth} MB`);
  if (process.env.STRICT_RSS === '1' && growth > 100) {
    console.error('FAIL: steady-state RSS grew > 100 MB');
    process.exit(1);
  }
}

async function cancellation(tts) {
  for (const [label, req] of [
         ['withConfig', {text: TEXT, generationConfig: {sid: 0, speed: 1.0}}],
         ['legacy', {text: TEXT, sid: 0, speed: 1.0}],
  ]) {
    let calls = 0;
    const r = await tts.generateAsync({
      ...req,
      onProgress: () => {
        calls++;
        return 0;  // cancel on first chunk
      },
    });
    await new Promise((res) => setTimeout(res, 200));
    if (calls !== 1) {
      console.error(
          `FAIL(${label}): expected exactly 1 callback after cancel, got ${calls}`);
      process.exit(1);
    }
    console.log(
        `cancellation (${label}): exactly 1 callback, samples: ${r.samples.length}`);
  }
}

async function throwing(tts) {
  // Contract: a throwing onProgress must not unwind through the N-API
  // boundary; the addon cancels the generation and the promise REJECTS with
  // the thrown error. Nothing may reach uncaughtException.
  let uncaught = null;
  const handler = (err) => {
    uncaught = err;
  };
  process.on('uncaughtException', handler);
  let calls = 0;
  let rejection = null;
  try {
    await tts.generateAsync({
      text: TEXT,
      generationConfig: {sid: 0, speed: 1.0},
      onProgress: () => {
        calls++;
        throw new Error('callback boom');
      },
    });
  } catch (err) {
    rejection = err;
  }
  await new Promise((res) => setTimeout(res, 200));
  process.removeListener('uncaughtException', handler);
  if (uncaught !== null) {
    console.error(`FAIL: throwing callback leaked to uncaughtException: ${
        uncaught.message}`);
    process.exit(1);
  }
  if (rejection === null ||
      !String(rejection.message).includes('callback boom')) {
    console.error(`FAIL: expected rejection carrying the thrown error, got ${
        rejection === null ? 'resolution' : rejection.message}`);
    process.exit(1);
  }
  if (calls !== 1) {
    console.error(
        `FAIL: expected exactly 1 callback before cancellation, got ${calls}`);
    process.exit(1);
  }
  console.log(
      `throwing callback: cancelled generation, promise rejected with "${
          rejection.message}"`);
}

async function concurrent(tts) {
  // Cancellation is BEST-EFFORT: the producer checks the flag before starting
  // the next batch, but batches already synthesized/queued are not rolled
  // back, so on fast hardware the cancelled result may still contain most or
  // all of the audio. The hard guarantees tested here are callback
  // suppression (exactly one invocation) and per-worker isolation (the other
  // seven generations run all their batches). Sample counts are logged for
  // information only.
  const multi = 'First sentence to speak aloud. Second sentence follows here. ' +
      'And a third sentence closes it out.';
  const fullChunkCounts = Array.from({length: 7}, () => 0);
  let cancelledCalls = 0;
  const jobs = fullChunkCounts.map(
      (_, i) => tts.generateAsync({
        text: multi,
        generationConfig: {sid: 0, speed: 1.0},
        onProgress: () => {
          fullChunkCounts[i]++;
          return 1;
        },
      }));
  jobs.push(tts.generateAsync({
    text: multi,
    generationConfig: {sid: 0, speed: 1.0},
    onProgress: () => {
      cancelledCalls++;
      return 0;
    },
  }));
  const results = await Promise.all(jobs);
  const full = results.slice(0, 7).map((r) => r.samples.length);
  const cancelled = results[7].samples.length;
  const minFull = Math.min(...full);
  const minChunks = Math.min(...fullChunkCounts);
  if (cancelledCalls !== 1) {
    console.error(`FAIL: concurrent cancel got ${cancelledCalls} callbacks`);
    process.exit(1);
  }
  if (minChunks < 2) {
    console.error(`FAIL: a full generation saw only ${
        minChunks} callbacks — cancellation leaked across workers`);
    process.exit(1);
  }
  console.log(`concurrent: 7 full (min ${minChunks} chunks, min ${
      minFull} samples) + 1 cancelled (${
      cancelled} samples, best-effort truncation), no leakage`);
}

async function copyBuffer(tts) {
  // Exercise the enableExternalBuffer:false marshalling branch
  let chunks = 0;
  const r = await tts.generateAsync({
    text: TEXT,
    generationConfig: {sid: 0, speed: 1.0},
    enableExternalBuffer: false,
    onProgress: () => {
      chunks++;
      return 1;
    },
  });
  if (!r.samples || r.samples.length === 0) {
    console.error('FAIL: enableExternalBuffer:false returned no samples');
    process.exit(1);
  }
  console.log(`copy-buffer: ${chunks} chunks, ${r.samples.length} samples`);
}

async function main() {
  const tts = makeTts();
  await sequential(tts);
  await cancellation(tts);
  await throwing(tts);
  await copyBuffer(tts);
  await concurrent(tts);
  console.log('PASS');
}

main().catch((err) => {
  console.error('FAIL:', err);
  process.exit(1);
});
