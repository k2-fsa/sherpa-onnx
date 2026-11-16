// Copyright (c)  2026  Xiaomi Corporation
'use strict'

const sherpa_onnx_asr = require('./sherpa-onnx-asr.js');

function freeConfig(config, Module) {
  if ('buffer' in config) {
    Module._free(config.buffer);
  }

  Module._free(config.ptr);
}

function initForcedAlignerConfig(config, Module) {
  const modelDir = config.modelDir || '';
  const modelDirLen = Module.lengthBytesUTF8(modelDir) + 1;
  const lexicon = config.lexicon || '';
  const lexiconLen = Module.lengthBytesUTF8(lexicon) + 1;

  const buffer = Module._malloc(modelDirLen + lexiconLen);

  const len = 2 * 4 + 5 * 4 + 4;
  const ptr = Module._malloc(len);

  let offset = 0;
  Module.stringToUTF8(modelDir, buffer + offset, modelDirLen);
  offset += modelDirLen;
  Module.stringToUTF8(lexicon, buffer + offset, lexiconLen);

  // A value of 0 lets the C API use its default for that option.
  const options = config.options || {};

  offset = 0;
  Module.setValue(ptr + offset, buffer, 'i8*');
  offset += 4;
  Module.setValue(ptr + offset, buffer + modelDirLen, 'i8*');
  offset += 4;
  Module.setValue(ptr + offset, options.beam || 0, 'float');
  offset += 4;
  Module.setValue(ptr + offset, options.retryBeam || 0, 'float');
  offset += 4;
  Module.setValue(ptr + offset, options.transitionScale || 0, 'float');
  offset += 4;
  Module.setValue(ptr + offset, options.selfLoopScale || 0, 'float');
  offset += 4;
  Module.setValue(ptr + offset, options.acousticScale || 0, 'float');
  offset += 4;
  Module.setValue(ptr + offset, config.debug || 0, 'i32');

  return {
    buffer: buffer,
    ptr: ptr,
    len: len,
  };
}

function readResult(ptr, Module) {
  const ans = {duration: 0, textgrid: '', tiers: []};
  if (!ptr) {
    return ans;
  }

  ans.duration = Module.getValue(ptr, 'float');
  ans.textgrid = Module.UTF8ToString(Module.getValue(ptr + 4, 'i8*'));

  const numTiers = Module.getValue(ptr + 8, 'i32');
  const tiersPtr = Module.getValue(ptr + 12, 'i8*');
  for (let i = 0; i < numTiers; ++i) {
    const tierPtr = tiersPtr + i * 12;
    const tier = {
      name: Module.UTF8ToString(Module.getValue(tierPtr, 'i8*')),
      intervals: [],
    };

    const numIntervals = Module.getValue(tierPtr + 4, 'i32');
    const intervalsPtr = Module.getValue(tierPtr + 8, 'i8*');
    for (let k = 0; k < numIntervals; ++k) {
      const intervalPtr = intervalsPtr + k * 12;
      tier.intervals.push({
        start: Module.getValue(intervalPtr, 'float'),
        end: Module.getValue(intervalPtr + 4, 'float'),
        text: Module.UTF8ToString(Module.getValue(intervalPtr + 8, 'i8*')),
      });
    }

    ans.tiers.push(tier);
  }

  return ans;
}

class ForcedAligner {
  constructor(configObj, Module) {
    const config = initForcedAlignerConfig(configObj, Module);
    const handle = Module._SherpaOnnxCreateForcedAligner(config.ptr);

    freeConfig(config, Module);

    if (!handle) {
      throw new Error('Failed to create the forced aligner');
    }

    this.handle = handle;
    this.Module = Module;
  }

  free() {
    if (this.handle) {
      this.Module._SherpaOnnxDestroyForcedAligner(this.handle);
      this.handle = 0;
    }
  }

  createStream() {
    const handle =
        this.Module._SherpaOnnxForcedAlignerCreateStream(this.handle);
    return new sherpa_onnx_asr.OfflineStream(handle, this.Module);
  }

  /**
   * @param stream {OfflineStream} Created by createStream()
   * @param text {String} The transcript of the whole utterance
   * @returns {Object} {duration, textgrid, tiers: [{name, intervals:
   *                   [{start, end, text}]}]}; empty for an empty result.
   *                   An interval has an empty text for a silence gap in
   *                   the "words" tier.
   */
  compute(stream, text) {
    const textLen = this.Module.lengthBytesUTF8(text) + 1;
    const textPtr = this.Module._malloc(textLen);
    this.Module.stringToUTF8(text, textPtr, textLen);

    const resultPtr = this.Module._SherpaOnnxForcedAlignerCompute(
        this.handle, stream.handle, textPtr);
    this.Module._free(textPtr);

    const ans = readResult(resultPtr, this.Module);
    this.Module._SherpaOnnxDestroyForcedAlignmentResult(resultPtr);
    return ans;
  }
}

module.exports = {
  ForcedAligner,
};
