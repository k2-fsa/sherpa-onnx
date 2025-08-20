// This file copies and modifies code
// from https://mdn.github.io/web-dictaphone/scripts/app.js
// and https://gist.github.com/meziantou/edb7217fddfbb70e899e

const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const clearBtn = document.getElementById('clearBtn');
const soundClips = document.getElementById('sound-clips');

let textArea = document.getElementById('results');

let lastResult = '';
let resultList = [];

clearBtn.onclick = function() {
  resultList = [];
  textArea.value = getDisplayResult();
  textArea.scrollTop = textArea.scrollHeight;  // auto scroll
};

function getDisplayResult() {
  let i = 0;
  let ans = '';
  for (let s in resultList) {
    if (resultList[s] == '') {
      continue;
    }

    if (resultList[s] == 'Speech detected') {
      ans += '' + i + ': ' + resultList[s];
      i += 1;
    } else {
      ans += ', ' + resultList[s] + '\n';
    }
  }

  if (lastResult.length > 0) {
    ans += '' + i + ': ' + lastResult + '\n';
  }
  return ans;
}

Module = {};

let audioCtx;
let mediaStream;

let expectedSampleRate = 16000;
let recordSampleRate;  // the sampleRate of the microphone
let recorder = null;   // the microphone
let leftchannel = [];  // TODO: Use a single channel

let recordingLength = 0;  // number of samples so far

let vad = null;
let buffer = null;
let recognizer = null;
let printed = false;

function fileExists(filename) {
  const filenameLen = Module.lengthBytesUTF8(filename) + 1;
  const buffer = Module._malloc(filenameLen);
  Module.stringToUTF8(filename, buffer, filenameLen);

  let exists = Module._SherpaOnnxFileExists(buffer);

  Module._free(buffer);

  return exists;
}

function initOfflineRecognizer() {
  let config = {
    modelConfig: {
      debug: 1,
      tokens: './tokens.txt',
    },
  };
  if (fileExists('sense-voice.onnx') == 1) {
    config.modelConfig.senseVoice = {
      model: './sense-voice.onnx',
      useInverseTextNormalization: 1,
    };
  } else if (fileExists('whisper-encoder.onnx')) {
    config.modelConfig.whisper = {
      encoder: './whisper-encoder.onnx',
      decoder: './whisper-decoder.onnx',
    };
  } else if (fileExists('transducer-encoder.onnx')) {
    config.modelConfig.transducer = {
      encoder: './transducer-encoder.onnx',
      decoder: './transducer-decoder.onnx',
      joiner: './transducer-joiner.onnx',
    };
    config.modelConfig.modelType = 'transducer';
  } else if (fileExists('nemo-transducer-encoder.onnx')) {
    config.modelConfig.transducer = {
      encoder: './nemo-transducer-encoder.onnx',
      decoder: './nemo-transducer-decoder.onnx',
      joiner: './nemo-transducer-joiner.onnx',
    };
    config.modelConfig.modelType = 'nemo_transducer';
  } else if (fileExists('paraformer.onnx')) {
    config.modelConfig.paraformer = {
      model: './paraformer.onnx',
    };
  } else if (fileExists('telespeech.onnx')) {
    config.modelConfig.telespeechCtc = './telespeech.onnx';
  } else if (fileExists('moonshine-preprocessor.onnx')) {
    config.modelConfig.moonshine = {
      preprocessor: './moonshine-preprocessor.onnx',
      encoder: './moonshine-encoder.onnx',
      uncachedDecoder: './moonshine-uncached-decoder.onnx',
      cachedDecoder: './moonshine-cached-decoder.onnx'
    };
  } else if (fileExists('dolphin.onnx')) {
    config.modelConfig.dolphin = {model: './dolphin.onnx'};
  } else if (fileExists('zipformer-ctc.onnx')) {
    // you need to rename model.int8.onnx from zipformer CTC to
    // zipformer-ctc.onnx
    config.modelConfig.zipformerCtc = {model: './zipformer-ctc.onnx'};
  } else {
    console.log('Please specify a model.');
    alert('Please specify a model.');
  }

  recognizer = new OfflineRecognizer(config, Module);
}

// https://emscripten.org/docs/api_reference/module.html#Module.locateFile
Module.locateFile = function(path, scriptDirectory = '') {
  console.log(`path: ${path}, scriptDirectory: ${scriptDirectory}`);
  return scriptDirectory + path;
};

// https://emscripten.org/docs/api_reference/module.html#Module.locateFile
Module.setStatus = function(status) {
  console.log(`status ${status}`);
  const statusElement = document.getElementById('status');
  if (status == 'Running...') {
    status = 'Model downloaded. Initializing recongizer...'
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

Module.onRuntimeInitialized = function() {
  console.log('inited!');

  startBtn.disabled = false;

  vad = createVad(Module);
  console.log('vad is created!', vad);

  buffer = new CircularBuffer(30 * 16000, Module);
  console.log('CircularBuffer is created!', buffer);

  initOfflineRecognizer();
};

if (navigator.mediaDevices.getUserMedia) {
  console.log('getUserMedia supported.');

  // see https://w3c.github.io/mediacapture-main/#dom-mediadevices-getusermedia
  const constraints = {audio: true};

  let onSuccess = function(stream) {
    if (!audioCtx) {
      audioCtx = new AudioContext({sampleRate: expectedSampleRate});
    }
    console.log(audioCtx);
    recordSampleRate = audioCtx.sampleRate;
    console.log('sample rate ' + recordSampleRate);

    // creates an audio node from the microphone incoming stream
    mediaStream = audioCtx.createMediaStreamSource(stream);
    console.log('media stream', mediaStream);

    // https://developer.mozilla.org/en-US/docs/Web/API/AudioContext/createScriptProcessor
    // bufferSize: the onaudioprocess event is called when the buffer is full
    var bufferSize = 4096;
    var numberOfInputChannels = 1;
    var numberOfOutputChannels = 2;
    if (audioCtx.createScriptProcessor) {
      recorder = audioCtx.createScriptProcessor(
          bufferSize, numberOfInputChannels, numberOfOutputChannels);
    } else {
      recorder = audioCtx.createJavaScriptNode(
          bufferSize, numberOfInputChannels, numberOfOutputChannels);
    }
    console.log('recorder', recorder);

    recorder.onaudioprocess = function(e) {
      let samples = new Float32Array(e.inputBuffer.getChannelData(0))
      samples = downsampleBuffer(samples, expectedSampleRate);
      buffer.push(samples);
      while (buffer.size() > vad.config.sileroVad.windowSize) {
        const s = buffer.get(buffer.head(), vad.config.sileroVad.windowSize);
        vad.acceptWaveform(s);
        buffer.pop(vad.config.sileroVad.windowSize);

        if (vad.isDetected() && !printed) {
          printed = true;
          lastResult = 'Speech detected';
        }

        if (!vad.isDetected()) {
          printed = false;
          if (lastResult != '') {
            resultList.push(lastResult);
          }
          lastResult = '';
        }

        while (!vad.isEmpty()) {
          const segment = vad.front();
          const duration = segment.samples.length / expectedSampleRate;
          let durationStr = `Duration: ${duration.toFixed(3)} seconds`;
          vad.pop();

          // non-streaming asr
          const stream = recognizer.createStream();
          stream.acceptWaveform(expectedSampleRate, segment.samples);
          recognizer.decode(stream);
          let recognitionResult = recognizer.getResult(stream);
          console.log(recognitionResult);
          let text = recognitionResult.text;
          stream.free();
          console.log(text);

          if (text != '') {
            durationStr += `. Result: ${text}`;
          }

          resultList.push(durationStr);

          // now save the segment to a wav file
          let buf = new Int16Array(segment.samples.length);
          for (var i = 0; i < segment.samples.length; ++i) {
            let s = segment.samples[i];
            if (s >= 1)
              s = 1;
            else if (s <= -1)
              s = -1;

            buf[i] = s * 32767;
          }

          let clipName = new Date().toISOString() + '--' + durationStr;

          const clipContainer = document.createElement('article');
          const clipLabel = document.createElement('p');
          const audio = document.createElement('audio');
          const deleteButton = document.createElement('button');

          clipContainer.classList.add('clip');
          audio.setAttribute('controls', '');
          deleteButton.textContent = 'Delete';
          deleteButton.className = 'delete';

          clipLabel.textContent = clipName;

          clipContainer.appendChild(audio);

          clipContainer.appendChild(clipLabel);
          clipContainer.appendChild(deleteButton);
          soundClips.appendChild(clipContainer);

          audio.controls = true;
          const blob = toWav(buf);

          leftchannel = [];
          const audioURL = window.URL.createObjectURL(blob);
          audio.src = audioURL;

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
      }

      textArea.value = getDisplayResult();
      textArea.scrollTop = textArea.scrollHeight;  // auto scroll
    };

    startBtn.onclick = function() {
      mediaStream.connect(recorder);
      recorder.connect(audioCtx.destination);

      console.log('recorder started');

      stopBtn.disabled = false;
      startBtn.disabled = true;
    };

    stopBtn.onclick = function() {
      vad.reset();
      buffer.reset();
      console.log('recorder stopped');

      // stopBtn recording
      recorder.disconnect(audioCtx.destination);
      mediaStream.disconnect(recorder);

      startBtn.style.background = '';
      startBtn.style.color = '';
      // mediaRecorder.requestData();

      stopBtn.disabled = true;
      startBtn.disabled = false;
    };
  };

  let onError = function(err) {
    console.log('The following error occured: ' + err);
  };

  navigator.mediaDevices.getUserMedia(constraints).then(onSuccess, onError);
} else {
  console.log('getUserMedia not supported on your browser!');
  alert('getUserMedia not supported on your browser!');
}

// this function is copied/modified from
// https://gist.github.com/meziantou/edb7217fddfbb70e899e
function flatten(listOfSamples) {
  let n = 0;
  for (let i = 0; i < listOfSamples.length; ++i) {
    n += listOfSamples[i].length;
  }
  let ans = new Int16Array(n);

  let offset = 0;
  for (let i = 0; i < listOfSamples.length; ++i) {
    ans.set(listOfSamples[i], offset);
    offset += listOfSamples[i].length;
  }
  return ans;
}

// this function is copied/modified from
// https://gist.github.com/meziantou/edb7217fddfbb70e899e
function toWav(samples) {
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
  view.setUint32(24, expectedSampleRate, true);  // sampleRate
  view.setUint32(28, expectedSampleRate * 2, true);  // byteRate
  view.setUint16(32, 2, true);                       // blockAlign
  view.setUint16(34, 16, true);                      // bitsPerSample
  view.setUint32(36, 0x61746164, true);              // Subchunk2ID
  view.setUint32(40, samples.length * 2, true);      // subchunk2Size

  let offset = 44;
  for (let i = 0; i < samples.length; ++i) {
    view.setInt16(offset, samples[i], true);
    offset += 2;
  }

  return new Blob([view], {type: 'audio/wav'});
}

// this function is copied from
// https://github.com/awslabs/aws-lex-browser-audio-capture/blob/master/lib/worker.js#L46
function downsampleBuffer(buffer, exportSampleRate) {
  if (exportSampleRate === recordSampleRate) {
    return buffer;
  }
  var sampleRateRatio = recordSampleRate / exportSampleRate;
  var newLength = Math.round(buffer.length / sampleRateRatio);
  var result = new Float32Array(newLength);
  var offsetResult = 0;
  var offsetBuffer = 0;
  while (offsetResult < result.length) {
    var nextOffsetBuffer = Math.round((offsetResult + 1) * sampleRateRatio);
    var accum = 0, count = 0;
    for (var i = offsetBuffer; i < nextOffsetBuffer && i < buffer.length; i++) {
      accum += buffer[i];
      count++;
    }
    result[offsetResult] = accum / count;
    offsetResult++;
    offsetBuffer = nextOffsetBuffer;
  }
  return result;
};
