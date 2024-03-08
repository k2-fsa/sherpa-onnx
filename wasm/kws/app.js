// This file copies and modifies code
// from https://mdn.github.io/web-dictaphone/scripts/app.js
// and https://gist.github.com/meziantou/edb7217fddfbb70e899e

const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const clearBtn = document.getElementById('clearBtn');
const hint = document.getElementById('hint');
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

    ans += '' + i + ': ' + resultList[s] + '\n';
    i += 1;
  }

  return ans;
}


Module = {};
Module.onRuntimeInitialized = function() {
  console.log('inited!');
  hint.innerText = 'Model loaded! Please click start';

  startBtn.disabled = false;

  recognizer = createKws(Module);
  console.log('recognizer is created!', recognizer);
};

let audioCtx;
let mediaStream;

let expectedSampleRate = 16000;
let recordSampleRate;  // the sampleRate of the microphone
let recorder = null;   // the microphone
let leftchannel = [];  // TODO: Use a single channel

let recordingLength = 0;  // number of samples so far

let recognizer = null;
let recognizer_stream = null;

if (navigator.mediaDevices.getUserMedia) {
  console.log('getUserMedia supported.');

  // see https://w3c.github.io/mediacapture-main/#dom-mediadevices-getusermedia
  const constraints = {audio: true};

  let onSuccess = function(stream) {
    if (!audioCtx) {
      audioCtx = new AudioContext({sampleRate: 16000});
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

      if (recognizer_stream == null) {
        recognizer_stream = recognizer.createStream();
      }

      recognizer_stream.acceptWaveform(expectedSampleRate, samples);
      while (recognizer.isReady(recognizer_stream)) {
        recognizer.decode(recognizer_stream);
      }


      let result = recognizer.getResult(recognizer_stream);
      console.log(result)

      if (result.keyword.length > 0) {
        lastResult = result;
        resultList.push(JSON.stringify(result));
      }


      textArea.value = getDisplayResult();
      textArea.scrollTop = textArea.scrollHeight;  // auto scroll

      let buf = new Int16Array(samples.length);
      for (var i = 0; i < samples.length; ++i) {
        let s = samples[i];
        if (s >= 1)
          s = 1;
        else if (s <= -1)
          s = -1;

        samples[i] = s;
        buf[i] = s * 32767;
      }

      leftchannel.push(buf);
      recordingLength += bufferSize;
    };

    startBtn.onclick = function() {
      mediaStream.connect(recorder);
      recorder.connect(audioCtx.destination);

      console.log('recorder started');

      stopBtn.disabled = false;
      startBtn.disabled = true;
    };

    stopBtn.onclick = function() {
      console.log('recorder stopped');

      // stopBtn recording
      recorder.disconnect(audioCtx.destination);
      mediaStream.disconnect(recorder);

      startBtn.style.background = '';
      startBtn.style.color = '';
      // mediaRecorder.requestData();

      stopBtn.disabled = true;
      startBtn.disabled = false;

      var clipName = new Date().toISOString();

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
      let samples = flatten(leftchannel);
      const blob = toWav(samples);

      leftchannel = [];
      const audioURL = window.URL.createObjectURL(blob);
      audio.src = audioURL;
      console.log('recorder stopped');

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