// return an object
// {
//   samples: a float32 array
//   sampleRate: an integer
// }
function readWave(filename, Module) {
  const filenameLen = Module.lengthBytesUTF8(filename) + 1;
  const pFilename = Module._malloc(filenameLen);
  Module.stringToUTF8(filename, pFilename, filenameLen);

  const w = Module._SherpaOnnxReadWave(pFilename);
  Module._free(pFilename);


  const samplesPtr = Module.HEAP32[w / 4] / 4;
  const sampleRate = Module.HEAP32[w / 4 + 1];
  const numSamples = Module.HEAP32[w / 4 + 2];

  const samples = new Float32Array(numSamples);
  for (let i = 0; i < numSamples; i++) {
    samples[i] = Module.HEAPF32[samplesPtr + i];
  }

  Module._SherpaOnnxFreeWave(w);


  return {samples: samples, sampleRate: sampleRate};
}

// data is an object
// {
//   samples: a float32 array
//   sampleRate: an integer
// }
function writeWave(filename, data, Module) {
  const pSamples =
      Module._malloc(data.samples.length * data.samples.BYTES_PER_ELEMENT);
  Module.HEAPF32.set(data.samples, pSamples / data.samples.BYTES_PER_ELEMENT);

  const filenameLen = Module.lengthBytesUTF8(filename) + 1;
  const pFilename = Module._malloc(filenameLen);
  Module.stringToUTF8(filename, pFilename, filenameLen);

  Module._SherpaOnnxWriteWave(
      pSamples, data.samples.length, data.sampleRate, pFilename);

  Module._free(pFilename);
  Module._free(pSamples);
}

if (typeof process == 'object' && typeof process.versions == 'object' &&
    typeof process.versions.node == 'string') {
  module.exports = {
    readWave,
    writeWave,
  };
}
