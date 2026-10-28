// Copyright (c)  2026  Nexmoe

const assert = require('assert');
const fs = require('fs');
const {createRequire} = require('module');
const {MessageChannel} = require('worker_threads');
const modulePath = process.argv[3] || require.resolve('../lib/sherpa-onnx.js');
const sherpa_onnx = require(modulePath);
const addon = createRequire(modulePath)('./addon.js');

/** Validate every native entry point without loading a model. */
function testValidation() {
  assert.strictEqual(typeof sherpa_onnx.OfflineSourceSeparation, 'function');
  for (const config of [undefined, null, true, 1, 'model', []]) {
    assert.throws(
        () => new sherpa_onnx.OfflineSourceSeparation(config), TypeError);
  }
  assert.throws(() => addon.createOfflineSourceSeparation(), TypeError);
  assert.throws(
      () => new sherpa_onnx.OfflineSourceSeparation({model: {}}),
      /source separation config/);

  for (const numThreads of [0, -1, NaN, Infinity, 1.5, 2 ** 31]) {
    assert.throws(
        () => new sherpa_onnx.OfflineSourceSeparation({model: {numThreads}}),
        RangeError);
  }
  assert.throws(
      () => new sherpa_onnx.OfflineSourceSeparation({model: {numThreads: '1'}}),
      TypeError);
  const getterError = new Error('Config getter failed');
  assert.throws(() => new sherpa_onnx.OfflineSourceSeparation({
    model: {
      spleeter: {
        vocals: 'partially-allocated.onnx',
        get accompaniment() { throw getterError; },
      },
    },
  }), error => error === getterError);

  const foreignHandle = addon.createLinearResampler(16000, 44100);
  for (const handle of [undefined, null, {}, 1, foreignHandle]) {
    assert.throws(
        () => addon.offlineSourceSeparationGetOutputSampleRate(handle),
        TypeError);
    assert.throws(
        () => addon.offlineSourceSeparationGetNumberOfStems(handle), TypeError);
    assert.throws(
        () => addon.offlineSourceSeparationProcess(handle, {}), TypeError);
  }
  assert.throws(
      () => addon.offlineSourceSeparationGetOutputSampleRate(), TypeError);
  assert.throws(
      () => addon.offlineSourceSeparationGetNumberOfStems(), TypeError);
  assert.throws(() => addon.offlineSourceSeparationProcess(), TypeError);
}

/** Round-trip multi-channel wave IO without loading a model. */
function testWaveIO() {
  assert.strictEqual(typeof sherpa_onnx.readWaveMultiChannel, 'function');
  assert.strictEqual(typeof sherpa_onnx.writeWaveMultiChannel, 'function');

  const left = new Float32Array(1600);
  const right = new Float32Array(1600);
  for (let i = 0; i < left.length; ++i) {
    left[i] = 0.9 * Math.sin(i / 10);
    right[i] = -0.5 * Math.cos(i / 7);
  }

  const filename = 'test-wave-io-roundtrip.wav';
  try {
    assert.strictEqual(
        sherpa_onnx.writeWaveMultiChannel(
            filename, {samples: [left, right], sampleRate: 16000}),
        true);
    for (const enableExternalBuffer of [true, false]) {
      const wave =
          sherpa_onnx.readWaveMultiChannel(filename, enableExternalBuffer);
      assert.strictEqual(wave.sampleRate, 16000);
      assert.deepStrictEqual(
          wave.samples.map(channel => channel.length), [1600, 1600]);
      // 16-bit PCM quantization
      for (const [actual, expected] of [[wave.samples[0], left],
                                        [wave.samples[1], right]]) {
        for (let i = 0; i < expected.length; ++i) {
          assert(Math.abs(actual[i] - expected[i]) <= 2 / 32768);
        }
      }
      assert.notDeepStrictEqual(wave.samples[0], wave.samples[1]);
    }

    assert.strictEqual(
        sherpa_onnx.writeWaveMultiChannel(
            filename, {samples: [left], sampleRate: 16000}),
        true);
    const mono = sherpa_onnx.readWaveMultiChannel(filename, false);
    assert.strictEqual(mono.samples.length, 1);
    assert.strictEqual(mono.samples[0].length, 1600);

    // Any channel count is fine, not only 1 or 2.
    assert.strictEqual(
        sherpa_onnx.writeWaveMultiChannel(
            filename, {samples: Array(3).fill(left), sampleRate: 16000}),
        true);
    const three = sherpa_onnx.readWaveMultiChannel(filename, false);
    assert.strictEqual(three.samples.length, 3);

    // A wave with an empty data chunk has no samples per channel.
    assert.strictEqual(
        sherpa_onnx.writeWave(
            filename, {samples: new Float32Array(0), sampleRate: 16000}),
        true);
    for (const enableExternalBuffer of [true, false]) {
      const empty =
          sherpa_onnx.readWaveMultiChannel(filename, enableExternalBuffer);
      assert.strictEqual(empty.samples.length, 1);
      assert.strictEqual(empty.samples[0].length, 0);
    }
  } finally {
    fs.unlinkSync(filename);
  }

  for (const obj of [undefined, null, 1, [], {samples: [left]},
                     {samples: [left], sampleRate: '16000'}]) {
    assert.throws(
        () => sherpa_onnx.writeWaveMultiChannel(filename, obj), TypeError);
  }
  for (const samples of [[], [new Float32Array(0)],
                         [new Float32Array(10), new Float32Array(11)]]) {
    assert.throws(
        () => sherpa_onnx.writeWaveMultiChannel(
            filename, {samples, sampleRate: 16000}),
        RangeError);
  }
  for (const samples of [[new Int16Array(10)], [left, 1], ['a']]) {
    assert.throws(
        () => sherpa_onnx.writeWaveMultiChannel(
            filename, {samples, sampleRate: 16000}),
        TypeError);
  }
  for (const sampleRate of [0, -1, NaN, Infinity, 44100.5, 2 ** 31]) {
    assert.throws(
        () => sherpa_onnx.writeWaveMultiChannel(
            filename, {samples: [left], sampleRate}),
        RangeError);
  }

  // A channel getter may detach an earlier channel's ArrayBuffer; the saved
  // pointers must not be read after any property getter has run.
  {
    const first = new Float32Array(10);
    const channels = [first];
    const {port1, port2} = new MessageChannel();
    try {
      Object.defineProperty(channels, 1, {
        get() {
          port1.postMessage(first.buffer, [first.buffer]);
          return new Float32Array(10);
        },
      });
      assert.throws(
          () => sherpa_onnx.writeWaveMultiChannel(
              filename, {samples: channels, sampleRate: 16000}),
          RangeError);
    } finally {
      port1.close();
      port2.close();
    }
  }
  assert.throws(() => sherpa_onnx.readWaveMultiChannel(), TypeError);
  assert.throws(() => sherpa_onnx.readWaveMultiChannel(1), TypeError);
  assert.throws(
      () => sherpa_onnx.readWaveMultiChannel('no-such-file.wav'), TypeError);
  assert.throws(
      () => sherpa_onnx.readWaveMultiChannel(filename, 1), TypeError);
  assert.throws(
      () => sherpa_onnx.readWaveMultiChannel(filename, true, true), TypeError);
}

/** Reject malformed audio before it reaches native inference. */
function testInvalidAudio(separator) {
  for (const request of [undefined, null, 1, [], {}]) {
    assert.throws(() => separator.process(request), TypeError);
  }
  const valid = {samples: [new Float32Array(44100)], sampleRate: 44100};
  for (const samples of [null, {}, new Float32Array(10), [0],
                        [new Int16Array(10)], [new Float64Array(10)]]) {
    assert.throws(() => separator.process({...valid, samples}), TypeError);
  }
  for (const samples of [[], [new Float32Array(0)],
                        [new Float32Array(10), new Float32Array(11)],
                        Array(3).fill(new Float32Array(10))]) {
    assert.throws(() => separator.process({...valid, samples}), RangeError);
  }
  for (const sampleRate of [undefined, '44100', null]) {
    assert.throws(() => separator.process({...valid, sampleRate}), TypeError);
  }
  for (const sampleRate of [0, -1, NaN, Infinity, 44100.5, 2 ** 31]) {
    assert.throws(() => separator.process({...valid, sampleRate}), RangeError);
  }
  assert.throws(
      () => separator.process({...valid, enableExternalBuffer: 1}), TypeError);
  for (const value of [NaN, Infinity, -Infinity]) {
    assert.throws(
        () => separator.process({...valid, samples: [new Float32Array([value])]}),
        RangeError);
  }
  assert.throws(
      () => addon.offlineSourceSeparationProcess(separator.handle), TypeError);

  const samples = new Float32Array(10);
  const {port1, port2} = new MessageChannel();
  try {
    port1.postMessage(samples.buffer, [samples.buffer]);
    assert.throws(
        () => separator.process({...valid, samples: [samples]}), RangeError);

    const first = new Float32Array(10);
    const channels = [first];
    Object.defineProperty(channels, 1, {
      get() {
        port1.postMessage(first.buffer, [first.buffer]);
        return new Float32Array(10);
      },
    });
    assert.throws(
        () => separator.process({...valid, samples: channels}), RangeError);
  } finally {
    port1.close();
    port2.close();
  }
}

/** Check stem shape, duration, finite samples, and non-silent output. */
function checkOutput(output, separator, duration) {
  assert.strictEqual(output.sampleRate, separator.sampleRate);
  assert.strictEqual(output.stems.length, separator.numStems);
  let energy = 0;
  for (const stem of output.stems) {
    assert.strictEqual(stem.length, 2);
    assert.strictEqual(stem[0].length, stem[1].length);
    for (const channel of stem) {
      assert(channel instanceof Float32Array);
      assert(channel.length > 0);
      assert(Math.abs(channel.length / output.sampleRate - duration) < 0.1);
      for (const sample of channel) {
        assert(Number.isFinite(sample));
        energy += sample * sample;
      }
    }
  }
  assert(energy > 0);
  assert.notDeepStrictEqual(output.stems[0], output.stems[1]);
}

/** Let native finalizers run while a single output channel remains live. */
async function collectGarbage() {
  assert.strictEqual(typeof global.gc, 'function', 'Run with --expose-gc');
  for (let i = 0; i < 3; ++i) {
    global.gc();
    await new Promise(resolve => setImmediate(resolve));
  }
}

/** Exercise real Spleeter or UVR inference through the freshly built addon. */
async function testModel(kind) {
  assert(['spleeter', 'uvr'].includes(kind), 'Expected spleeter or uvr');
  const model = kind === 'spleeter' ? {
    spleeter: {
      vocals: './sherpa-onnx-spleeter-2stems-fp16/vocals.fp16.onnx',
      accompaniment:
          './sherpa-onnx-spleeter-2stems-fp16/accompaniment.fp16.onnx',
    },
    numThreads: 1,
    provider: 'cpu',
    debug: false,
  } : {
    uvr: {model: './UVR-MDX-NET-Voc_FT.onnx'},
    numThreads: 1,
    provider: 'cpu',
    debug: 0,
  };
  let separator = new sherpa_onnx.OfflineSourceSeparation({model});
  assert.strictEqual(separator.numStems, 2);
  assert.strictEqual(separator.sampleRate, 44100);
  testInvalidAudio(separator);

  const wave = sherpa_onnx.readWave('./qi-feng-le-zh.wav', false);
  const n = Math.min(wave.samples.length, 3 * wave.sampleRate);
  // Nonzero byte offsets catch bindings that ignore TypedArray views.
  const left = new Float32Array(n + 4).subarray(1, n + 1);
  const right = new Float32Array(n + 6).subarray(3, n + 3);
  left.set(wave.samples.subarray(0, n));
  right.set(left.map(sample => -0.5 * sample));
  const pristine = [left.slice(), right.slice()];
  const request = {samples: [left, right], sampleRate: wave.sampleRate};

  let external = separator.process(request);
  checkOutput(external, separator, n / wave.sampleRate);
  let copied = separator.process({
    ...request, samples: pristine, enableExternalBuffer: false,
  });
  checkOutput(copied, separator, n / wave.sampleRate);
  assert.deepStrictEqual(copied, external);
  assert.deepStrictEqual([left, right], pristine);

  let mono = separator.process({...request, samples: [left]});
  let stereo = separator.process({...request, samples: [left, left]});
  assert.deepStrictEqual(mono, stereo);

  // The test wav is real stereo; readWaveMultiChannel must keep both channels
  // while readWave keeps only the first one.
  const multi = sherpa_onnx.readWaveMultiChannel('./qi-feng-le-zh.wav', false);
  assert.strictEqual(multi.sampleRate, wave.sampleRate);
  assert.deepStrictEqual(
      multi.samples.map(channel => channel.length),
      [wave.samples.length, wave.samples.length]);
  assert.deepStrictEqual(multi.samples[0], wave.samples);
  assert.notDeepStrictEqual(multi.samples[0], multi.samples[1]);
  checkOutput(
      separator.process({
        samples: multi.samples.map(channel => channel.subarray(0, n)),
        sampleRate: multi.sampleRate,
        enableExternalBuffer: false,
      }),
      separator, n / multi.sampleRate);

  const resampler = new sherpa_onnx.LinearResampler(wave.sampleRate, 16000);
  const resampled = resampler.flush(left);
  let resampledOutput = separator.process({
    samples: [resampled], sampleRate: 16000, enableExternalBuffer: false,
  });
  checkOutput(resampledOutput, separator, resampled.length / 16000);

  const retained = external.stems[1][1];
  const expected = retained.slice();
  const copiedChannel = copied.stems[0][0];
  const expectedCopy = copiedChannel.slice();
  separator = external = copied = mono = stereo = resampledOutput = null;
  await collectGarbage();
  assert.deepStrictEqual(retained, expected);
  assert.deepStrictEqual(copiedChannel, expectedCopy);
  console.log(`${kind}: inference, validation, mono/stereo, buffers, resampling, and GC passed`);
}

/** Run model-free checks, then optional real-model regression checks. */
async function main() {
  testValidation();
  testWaveIO();
  if (process.argv[2]) await testModel(process.argv[2]);
  console.log('Source separation tests passed');
}

main().catch(error => {
  console.error(error);
  process.exitCode = 1;
});
