const generateBtn = document.getElementById('generateBtn');
const speakerIdLabel = document.getElementById('speakerIdLabel');
const speakerIdInput = document.getElementById('speakerId');
const speakerIdSection = document.getElementById('speakerIdSection');
const langSection = document.getElementById('langSection');
const langSelect = document.getElementById('lang');
const frontendSection = document.getElementById('frontendSection');
const frontendPhonemizeInput = document.getElementById('frontendPhonemize');
const frontendLexiconInput = document.getElementById('frontendLexicon');
const referenceAudioSection = document.getElementById('referenceAudioSection');
const referenceTextSection = document.getElementById('referenceTextSection');
const referenceAudioInput = document.getElementById('referenceAudio');
const referenceTextInput = document.getElementById('referenceText');
const speedInput = document.getElementById('speed');
const speedValue = document.getElementById('speedValue');
const textArea = document.getElementById('text');
const soundClips = document.getElementById('sound-clips');
const statusElement = document.getElementById('status');
const generationStatusElement = document.getElementById('generationStatus');

speedValue.innerHTML = speedInput.value;

let index = 0;

let audioCtx = null;
const worker = new Worker("sherpa-onnx-tts.worker.js");
let ttsInstanceInfo = {
  modelType: null,
  usePhonemize: false,
  numSpeakers: 0,
  lang: '',
  isReady: false,
};
worker.onmessage = (e) => {
  if (e.data.type === "sherpa-onnx-tts-progress") {
    uiStatus.setStatus(e.data.status);
    return;
  }
  if (e.data.type === "sherpa-onnx-tts-generation-progress") {
    const percent = Math.max(0, Math.min(100, (e.data.progress || 0) * 100));
    setGenerationStatus(`Generating audio... ${percent.toFixed(2)}%`);
    return;
  }
  if (e.data.type === "sherpa-onnx-tts-ready") {
    ttsInstanceInfo.modelType = e.data.modelType;
    ttsInstanceInfo.usePhonemize = e.data.usePhonemize;
    ttsInstanceInfo.numSpeakers = e.data.numSpeakers;
    ttsInstanceInfo.lang = e.data.lang;
    ttsInstanceInfo.isReady = true;
    generateBtn.disabled = false;
    speakerIdLabel.innerHTML = `Speaker ID (0 - ${e.data.numSpeakers - 1}):`;
    updateUiForModelType();
    uiStatus.setStatus('');
    setGenerationStatus('');
    return;
  }
  if (e.data.type === "error") {
    generateBtn.disabled = false;
    if (ttsInstanceInfo.isReady) {
      setGenerationStatus(e.data.message);
    } else {
      uiStatus.setStatus(e.data.message);
    }
    return;
  }
  if (e.data.type === "sherpa-onnx-tts-result") {
    let audio = e.data;
    generateBtn.disabled = false;
    setGenerationStatus('');

    console.log(audio.samples.length, audio.sampleRate);

    if (!audioCtx) {
      audioCtx = new AudioContext({ sampleRate: audio.sampleRate });
    }

    const buffer = audioCtx.createBuffer(
      1,
      audio.samples.length,
      audio.sampleRate,
    );

    buffer.getChannelData(0).set(audio.samples); // 使用 .set() 比 for 循环快得多
    const source = audioCtx.createBufferSource();
    source.buffer = buffer;
    source.connect(audioCtx.destination);
    source.start();

    createAudioTag(audio);
  }
};

// UI status helper, used by the sherpa-onnx worker messages below.
const uiStatus = {};

// Mirrors emscripten's Module.setStatus:
// https://emscripten.org/docs/api_reference/module.html#Module.setStatus
uiStatus.setStatus = function(status) {
  console.log(`status ${status}`);
  if (status == 'Running...') {
    status = 'Model downloaded. Initializing text to speech model...'
  }

  const downloadMatch = status.match(/Downloading data... \((\d+)\/(\d+)\)/);
  if (downloadMatch) {
    const downloaded = BigInt(downloadMatch[1]);
    const total = BigInt(downloadMatch[2]);
    const percent =
        total === 0 ? 0.00 : Number((downloaded * 10000n) / total) / 100;
    const downloadedMB = Number(downloaded) / (1024 * 1024);
    const totalMB = Number(total) / (1024 * 1024);
    status = `Downloading data... ${percent.toFixed(2)}% (${downloadedMB.toFixed(2)} MB/${
        totalMB.toFixed(2)} MB)`;
    console.log(`here ${status}`)
  }

  statusElement.textContent = status;
  if (status === '') {
    statusElement.style.display = 'none';
    // statusElement.parentNode.removeChild(statusElement);

    document.querySelectorAll('.tab-content').forEach((tabContentElement) => {
      tabContentElement.classList.remove('loading');
    });
  } else {
    statusElement.style.display = 'block';
    document.querySelectorAll('.tab-content').forEach((tabContentElement) => {
      tabContentElement.classList.add('loading');
    });
  }
};

// piper-phonemize's Emscripten module. This mirrors its standalone demo
// (piper-phonemize/index.html + piper-phonemize/app-phonemize.js shipped in
// the bundle): the app defines `Module`, the glue script is loaded last (see
// index.html) and attaches its runtime to the same object.
//
// The one difference from that demo: it lives next to the bundle files while
// this page is one directory above them. When the glue fetches its .data it
// calls locateFile(name, "") with an empty scriptDirectory, so we must prefix
// "piper-phonemize/" ourselves - otherwise the .data request resolves against
// this page's URL, 404s, the espeak-ng data never reaches the wasm FS, and
// espeak init fails.
let resolvePiperReady;
let rejectPiperReady;
const piperReady = new Promise((resolve, reject) => {
  resolvePiperReady = resolve;
  rejectPiperReady = reject;
});
// Failures are reported at the generate step below; don't also warn here.
piperReady.catch(() => {});

// https://emscripten.org/docs/api_reference/module.html#Module.locateFile
var Module = {
  locateFile: function(path, scriptDirectory = '') {
    return (scriptDirectory || 'piper-phonemize/') + path;
  },

  setStatus: function(status) {
    // piper's runtime statuses are not about the TTS model; keep the status
    // bar for the sherpa-onnx download/init progress.
    if (status === 'Running...') return;
    uiStatus.setStatus(status);
  },

  onRuntimeInitialized: function() {
    console.log('piper-phonemize WASM runtime initialized');
    try {
      const ret = piperPhonemizeInitialize();
      console.log('piperPhonemizeInitialize returned:', ret);
      if (ret >= 0) {
        // 22050 on the first call, 0 if espeak-ng was already initialized
        resolvePiperReady();
      } else {
        rejectPiperReady(new Error('failed to initialize espeak-ng'));
      }
    } catch (e) {
      console.error('piper-phonemize initialization error:', e);
      rejectPiperReady(e);
    }
  },
};

// The glue is loaded after this file. If the bundle is missing entirely (see
// assets/README.md), say so instead of waiting forever on the promise above.
if (typeof piperPhonemizeInitialize !== 'function') {
  rejectPiperReady(new Error(
      'piper-phonemize bundle not loaded; see wasm/tts/assets/README.md'));
}

speedInput.oninput = function() {
  speedValue.innerHTML = this.value;
};

function updateUiForModelType() {
  const isZipVoice = ttsInstanceInfo.modelType === 4;
  const isPocketTts = ttsInstanceInfo.modelType === 5;
  const isSupertonic = ttsInstanceInfo.modelType === 7;
  const useGenerationConfig = isZipVoice || isPocketTts;
  speakerIdSection.classList.toggle('hidden', useGenerationConfig);
  referenceAudioSection.classList.toggle('hidden', !useGenerationConfig);
  referenceTextSection.classList.toggle('hidden', !isZipVoice);
  // SupertonicTTS supports 31 languages; no other model here needs a
  // language picker.
  langSection.classList.toggle('hidden', !isSupertonic);

  // The picker is only meaningful for the English piper-family models, which
  // can use either frontend. The Chinese/multi-lingual/zipvoice models have a
  // lexicon only, the pocket model has its own frontend, and the German model
  // is phonemize-only (no German lexicon exists).
  const isEnglishFamily = [0, 3, 6].includes(ttsInstanceInfo.modelType) &&
      (ttsInstanceInfo.lang || '').toLowerCase().startsWith('en');
  frontendSection.classList.toggle('hidden', !isEnglishFamily);
  frontendPhonemizeInput.checked = ttsInstanceInfo.usePhonemize;
  frontendLexiconInput.checked = !ttsInstanceInfo.usePhonemize;
}

function onFrontendChange() {
  const usePhonemize = frontendPhonemizeInput.checked;
  generateBtn.disabled = true;
  setGenerationStatus('Switching text frontend...');
  worker.postMessage({type: 'setFrontend', usePhonemize: usePhonemize});
}

frontendPhonemizeInput.addEventListener('change', onFrontendChange);
frontendLexiconInput.addEventListener('change', onFrontendChange);

function setGenerationStatus(status) {
  if (!generationStatusElement) {
    return;
  }

  generationStatusElement.textContent = status;
  generationStatusElement.style.display = status ? 'block' : 'none';
}

function getMonoSamples(audioBuffer) {
  if (audioBuffer.numberOfChannels === 1) {
    return new Float32Array(audioBuffer.getChannelData(0));
  }

  const samples = new Float32Array(audioBuffer.length);
  for (let c = 0; c < audioBuffer.numberOfChannels; ++c) {
    const channel = audioBuffer.getChannelData(c);
    for (let i = 0; i < channel.length; ++i) {
      samples[i] += channel[i];
    }
  }

  for (let i = 0; i < samples.length; ++i) {
    samples[i] /= audioBuffer.numberOfChannels;
  }

  return samples;
}

async function readReferenceAudio(file) {
  const arrayBuffer = await file.arrayBuffer();
  const ctx = new AudioContext();
  try {
    const audioBuffer = await ctx.decodeAudioData(arrayBuffer.slice(0));
    return {
      samples: getMonoSamples(audioBuffer),
      sampleRate: audioBuffer.sampleRate,
    };
  } finally {
    await ctx.close();
  }
}

function isWaveFile(file) {
  const name = file.name || '';
  return name.toLowerCase().endsWith('.wav');
}

function sanitizeFilename(name) {
  return name.replace(/[^a-zA-Z0-9._-]+/g, '-');
}

function downloadBlob(blob, filename) {
  const url = window.URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  window.URL.revokeObjectURL(url);
}

generateBtn.onclick = async function() {
  const isZipVoice = ttsInstanceInfo.modelType === 4;
  const isPocketTts = ttsInstanceInfo.modelType === 5;
  const isSupertonic = ttsInstanceInfo.modelType === 7;
  const useGenerationConfig = isZipVoice || isPocketTts;

  let speakerId = speakerIdInput.value;
  if (!useGenerationConfig) {
    if (speakerId.trim().length == 0) {
      alert('Please input a speakerId');
      return;
    }

    if (!speakerId.match(/^\d+$/)) {
      alert(`Input speakerID ${
          speakerId} is not a number.\nPlease enter a number between 0 and ${
          ttsInstanceInfo.numSpeakers - 1}`);
      return;
    }
    speakerId = parseInt(speakerId, 10);
    if (speakerId > ttsInstanceInfo.numSpeakers - 1) {
      alert(`Pleaser enter a number between 0 and ${ttsInstanceInfo.numSpeakers - 1}`);
      return;
    }
  }

  let text = textArea.value.trim();
  if (text.length == 0) {
    alert('Please input a non-blank text');
    return;
  }

  console.log('speakerId', speakerId);
  console.log('speed', speedInput.value);
  console.log('text', text);

  if (isSupertonic) {
    // SupertonicTTS has its own text frontend (unicode indexer): pass the
    // raw text plus the language; no reference audio and no phonemize.
    // numSteps is 8, like python-api-examples/supertonic-tts.py and
    // nodejs-examples/test-offline-tts-supertonic.js.
    generateBtn.disabled = true;
    setGenerationStatus('Generating audio...');
    worker.postMessage({
      text,
      type: 'generateWithConfig',
      genConfig: {
        sid: speakerId,
        speed: parseFloat(speedInput.value),
        numSteps: 8,
        extra: {lang: langSelect.value},
      },
    });
    return;
  }

  if (useGenerationConfig) {
    if (!referenceAudioInput.files || referenceAudioInput.files.length === 0) {
      alert('Please select a reference audio file');
      return;
    }

    const referenceFile = referenceAudioInput.files[0];
    if (!isWaveFile(referenceFile)) {
      alert('Please select a .wav reference audio file');
      return;
    }

    const referenceAudio = await readReferenceAudio(referenceFile);
    const genConfig = {
      speed: parseFloat(speedInput.value),
      referenceAudio: referenceAudio.samples,
      referenceSampleRate: referenceAudio.sampleRate,
      numSteps: isPocketTts ? 5 : 4,
    };

    if (isZipVoice) {
      const referenceText = referenceTextInput.value.trim();
      if (referenceText.length === 0) {
        alert('Please input the transcript of the reference audio');
        return;
      }

      genConfig.referenceText = referenceText;
      genConfig.extra = {
        min_char_in_sentence: 10,
      };
    }

    generateBtn.disabled = true;
    setGenerationStatus('Generating audio...');

    worker.postMessage({
      text,
      genConfig,
      type: "generateWithConfig",
    }, [genConfig.referenceAudio.buffer]);
    return;
  }

  if (ttsInstanceInfo.usePhonemize) {
    // Option 1: phonemize the text with piper-phonemize (loaded from
    // ./piper-phonemize/, see assets/README.md) and pass the phoneme
    // codepoints. No lexicon file and no espeak-ng-data are needed. Set
    // usePhonemize = false in sherpa-onnx-tts.js for option 2 (lexicon).
    try {
      await piperReady;
    } catch (e) {
      generateBtn.disabled = false;
      setGenerationStatus(`piper-phonemize is not available: ${e.message}`);
      return;
    }
    const voice = ttsInstanceInfo.lang || 'en-us';
    const sentences = piperPhonemize(text, voice);
    if (!sentences) {
      generateBtn.disabled = false;
      setGenerationStatus('Phonemization failed');
      return;
    }
    // piper may emit empty sentences (e.g. a trailing segment after
    // punctuation); the C API requires every sentence to be non-empty.
    const phonemeCodepoints = sentences.filter((sentence) => sentence.length > 0);
    generateBtn.disabled = true;
    setGenerationStatus('Generating audio...');
    worker.postMessage({
      text: '',
      genConfig: {
        phonemeCodepoints: phonemeCodepoints,
        sid: speakerId,
        speed: parseFloat(speedInput.value),
      },
      type: "generateWithConfig",
    });
    return;
  }

  worker.postMessage({
    text,
    sid: speakerId,
    speed: parseFloat(speedInput.value),
    type: "generate",
  });
};

function createAudioTag(generateAudio) {
  const blob = toWav(generateAudio.samples, generateAudio.sampleRate);

  const text = textArea.value.trim().substring(0, 100);
  const clipName = `${index} ${text} ...`;
  const filename = `${sanitizeFilename(clipName)}.wav`;
  index += 1;

  const clipContainer = document.createElement('article');
  const clipLabel = document.createElement('p');
  const audio = document.createElement('audio');
  const saveButton = document.createElement('button');
  const deleteButton = document.createElement('button');
  clipContainer.classList.add('clip');
  audio.setAttribute('controls', '');
  saveButton.textContent = 'Save';
  saveButton.className = 'save';
  deleteButton.textContent = 'Delete';
  deleteButton.className = 'delete';

  clipLabel.textContent = clipName;

  clipContainer.appendChild(audio);

  clipContainer.appendChild(clipLabel);
  clipContainer.appendChild(saveButton);
  clipContainer.appendChild(deleteButton);
  soundClips.appendChild(clipContainer);

  audio.controls = true;

  const audioURL = window.URL.createObjectURL(blob);
  audio.src = audioURL;

  saveButton.onclick = function() {
    downloadBlob(blob, filename);
  };

  deleteButton.onclick = function(e) {
    let evtTgt = e.target;
    evtTgt.parentNode.parentNode.removeChild(evtTgt.parentNode);
  };

  clipLabel.onclick = function() {
    const existingName = clipLabel.textContent;
    const newClipName = prompt('Enter a new name for your sound clip?');
    if (newClipName === null) {
      clipLabel.textContent = existingName;
    } else {
      clipLabel.textContent = newClipName;
    }
  };
}

// this function is copied/modified from
// https://gist.github.com/meziantou/edb7217fddfbb70e899e
function toWav(floatSamples, sampleRate) {
  let samples = new Int16Array(floatSamples.length);
  for (let i = 0; i < samples.length; ++i) {
    let s = floatSamples[i];
    if (s >= 1)
      s = 1;
    else if (s <= -1)
      s = -1;

    samples[i] = s * 32767;
  }

  let buf = new ArrayBuffer(44 + samples.length * 2);
  var view = new DataView(buf);

  // http://soundfile.sapp.org/doc/WaveFormat/
  //                   F F I R
  view.setUint32(0, 0x46464952, true);               // chunkID
  view.setUint32(4, 36 + samples.length * 2, true);  // chunkSize
  //                   E V A W
  view.setUint32(8, 0x45564157, true);  // format
                                        //
  //                      t m f
  view.setUint32(12, 0x20746d66, true);          // subchunk1ID
  view.setUint32(16, 16, true);                  // subchunk1Size, 16 for PCM
  view.setUint32(20, 1, true);                   // audioFormat, 1 for PCM
  view.setUint16(22, 1, true);                   // numChannels: 1 channel
  view.setUint32(24, sampleRate, true);          // sampleRate
  view.setUint32(28, sampleRate * 2, true);      // byteRate
  view.setUint16(32, 2, true);                   // blockAlign
  view.setUint16(34, 16, true);                  // bitsPerSample
  view.setUint32(36, 0x61746164, true);          // Subchunk2ID
  view.setUint32(40, samples.length * 2, true);  // subchunk2Size

  let offset = 44;
  for (let i = 0; i < samples.length; ++i) {
    view.setInt16(offset, samples[i], true);
    offset += 2;
  }

  return new Blob([view], {type: 'audio/wav'});
}
