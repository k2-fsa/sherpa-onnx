// Copyright (c)  2026  Xiaomi Corporation

const assert = require('assert');
const fs = require('fs');
const { MessageChannel } = require('worker_threads');
const sherpa_onnx = require('../lib/sherpa-onnx.js');
const addon = require('../lib/addon.js');

/** Round-trip mono readWave/writeWave and readWaveFromBinary. */
function testMonoWaveIO() {
    const samples = new Float32Array(1600);
    for (let i = 0; i < samples.length; ++i) {
        samples[i] = 0.5 * Math.sin(i / 10);
    }

    const filename = 'test-wave-io-mono.wav';
    try {
        assert.strictEqual(
            sherpa_onnx.writeWave(filename, { samples, sampleRate: 16000 }), true);
        for (const enableExternalBuffer of [true, false]) {
            const wave = sherpa_onnx.readWave(filename, enableExternalBuffer);
            assert.strictEqual(wave.sampleRate, 16000);
            assert.strictEqual(wave.samples.length, 1600);
            // 16-bit PCM quantization
            for (let i = 0; i < samples.length; ++i) {
                assert(Math.abs(wave.samples[i] - samples[i]) <= 2 / 32768);
            }
        }

        const data = new Uint8Array(fs.readFileSync(filename));
        // readWaveFromBinary is only exported by the raw addon.
        const fromBinary = addon.readWaveFromBinary(data, false);
        assert.strictEqual(fromBinary.sampleRate, 16000);
        assert.strictEqual(fromBinary.samples.length, 1600);
    } finally {
        fs.unlinkSync(filename);
    }
}

/** Round-trip multi-channel wave IO without loading a model. */
function testMultiChannelWaveIO() {
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
                filename, { samples: [left, right], sampleRate: 16000 }),
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
                filename, { samples: [left], sampleRate: 16000 }),
            true);
        const mono = sherpa_onnx.readWaveMultiChannel(filename, false);
        assert.strictEqual(mono.samples.length, 1);
        assert.strictEqual(mono.samples[0].length, 1600);

        // Any channel count is fine, not only 1 or 2.
        assert.strictEqual(
            sherpa_onnx.writeWaveMultiChannel(
                filename, { samples: Array(3).fill(left), sampleRate: 16000 }),
            true);
        const three = sherpa_onnx.readWaveMultiChannel(filename, false);
        assert.strictEqual(three.samples.length, 3);

        // A wave with an empty data chunk has no samples per channel.
        assert.strictEqual(
            sherpa_onnx.writeWave(
                filename, { samples: new Float32Array(0), sampleRate: 16000 }),
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

    for (const obj of [undefined, null, 1, [], { samples: [left] },
        { samples: [left], sampleRate: '16000' }]) {
        assert.throws(
            () => sherpa_onnx.writeWaveMultiChannel(filename, obj), TypeError);
    }
    for (const samples of [[], [new Float32Array(0)],
    [new Float32Array(10), new Float32Array(11)]]) {
        assert.throws(
            () => sherpa_onnx.writeWaveMultiChannel(
                filename, { samples, sampleRate: 16000 }),
            RangeError);
    }
    for (const samples of [[new Int16Array(10)], [left, 1], ['a']]) {
        assert.throws(
            () => sherpa_onnx.writeWaveMultiChannel(
                filename, { samples, sampleRate: 16000 }),
            TypeError);
    }
    for (const sampleRate of [0, -1, NaN, Infinity, 44100.5, 2 ** 31]) {
        assert.throws(
            () => sherpa_onnx.writeWaveMultiChannel(
                filename, { samples: [left], sampleRate }),
            RangeError);
    }

    // A channel getter may detach an earlier channel's ArrayBuffer; the saved
    // pointers must not be read after any property getter has run.
    {
        const first = new Float32Array(10);
        const channels = [first];
        const { port1, port2 } = new MessageChannel();
        try {
            Object.defineProperty(channels, 1, {
                get() {
                    port1.postMessage(first.buffer, [first.buffer]);
                    return new Float32Array(10);
                },
            });
            assert.throws(
                () => sherpa_onnx.writeWaveMultiChannel(
                    filename, { samples: channels, sampleRate: 16000 }),
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

function main() {
    testMonoWaveIO();
    testMultiChannelWaveIO();
    console.log('Wave IO tests passed');
}

main();
