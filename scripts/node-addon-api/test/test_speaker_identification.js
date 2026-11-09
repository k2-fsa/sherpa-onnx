// Copyright (c)  2026  Xiaomi Corporation

const assert = require('assert');
const { MessageChannel } = require('worker_threads');
const sherpa_onnx = require('../lib/sherpa-onnx.js');
const addon = require('../lib/addon.js');

const dim = 2;

/** Exercise add/verify/search/remove through the class API. */
function testHappyPath() {
    const manager = new sherpa_onnx.SpeakerEmbeddingManager(dim);
    const v = new Float32Array([0.1, 0.2]);

    assert.strictEqual(manager.add({ name: 'a', v }), true);
    assert.strictEqual(manager.contains('a'), true);
    assert.strictEqual(manager.getNumSpeakers(), 1);
    assert.deepStrictEqual(manager.getAllSpeakerNames(), ['a']);
    assert.strictEqual(manager.verify({ name: 'a', v, threshold: 0.5 }), true);
    assert.strictEqual(manager.search({ v, threshold: 0.5 }), 'a');

    assert.strictEqual(
        manager.addMulti(
            { name: 'b', v: [new Float32Array(dim), new Float32Array(dim)] }),
        true);
    assert.strictEqual(manager.getNumSpeakers(), 2);

    assert.strictEqual(manager.remove('a'), true);
    assert.strictEqual(manager.contains('a'), false);
}

/** Reject buffers that do not hold exactly dim elements. */
function testWrongLength() {
    const manager = new sherpa_onnx.SpeakerEmbeddingManager(dim);
    for (const v of [new Float32Array(0), new Float32Array(1),
    new Float32Array(3)]) {
        assert.throws(() => manager.add({ name: 'a', v }), RangeError);
        assert.throws(
            () => manager.verify({ name: 'a', v, threshold: 0.5 }), RangeError);
        assert.throws(() => manager.search({ v, threshold: 0.5 }), RangeError);
    }
}

/**
 * A property getter may detach the embedding's ArrayBuffer before its data
 * pointer is used; that must raise an error instead of crashing in native
 * code.
 */
function testDetachedBuffers() {
    const manager = new sherpa_onnx.SpeakerEmbeddingManager(dim);

    const cases = [
        (v, steal) => manager.add({ v, get name() { steal(); return 'a'; } }),
        (v, steal) => manager.search({ v, get threshold() { steal(); return 0.5; } }),
        (v, steal) => manager.verify(
            { name: 'a', v, get threshold() { steal(); return 0.5; } }),
        (v, steal) => addon.speakerEmbeddingManagerAddListFlattened(
            manager.handle, { vv: v, name: 'a', get n() { steal(); return 1; } }),
    ];

    for (const run of cases) {
        const { port1, port2 } = new MessageChannel();
        const v = new Float32Array([0.1, 0.2]);
        // A property may be read more than once; transfer the buffer only once.
        let stolen = false;
        const steal = () => {
            if (!stolen) {
                stolen = true;
                port1.postMessage(v.buffer, [v.buffer]);
            }
        };
        assert.throws(() => run(v, steal), RangeError);
        port1.close();
        port2.close();
    }
}

/** n must be positive and n * dim must match the flattened length. */
function testAddListFlattenedLength() {
    const manager = new sherpa_onnx.SpeakerEmbeddingManager(dim);
    const vv = new Float32Array(2 * dim);

    for (const n of [0, -1, 5]) {
        assert.throws(
            () => addon.speakerEmbeddingManagerAddListFlattened(
                manager.handle, { name: 'a', vv, n }),
            RangeError);
    }
    assert.strictEqual(
        addon.speakerEmbeddingManagerAddListFlattened(
            manager.handle, { name: 'a', vv, n: 2 }),
        true);
}

function main() {
    testHappyPath();
    testWrongLength();
    testDetachedBuffers();
    testAddListFlattenedLength();
    console.log('Speaker identification tests passed');
}

main();
