// This file copies and modifies code
// from https://mdn.github.io/web-dictaphone/scripts/app.js
// and https://gist.github.com/meziantou/edb7217fddfbb70e899e

var socket;
var recognition_text = [];

function getDisplayResult() {
  let i = 0;
  let ans = '';
  for (let s in recognition_text) {
    if (recognition_text[s] == '') continue;

    ans += '' + i + ': ' + recognition_text[s] + '\n';
    i += 1;
  }
  return ans;
}

function initWebSocket() {
  console.log('Creating websocket')
  let protocol = 'ws://';
  if (window.location.protocol == 'https:') {
    protocol = 'wss://'
  }
  let server_ip = serverIpInput.value;
  let server_port = serverPortInput.value;
  console.log('protocol: ', protocol);
  console.log('server_ip: ', server_ip);
  console.log('server_port: ', server_port);

  let uri = protocol + server_ip + ':' + server_port;
  console.log('uri', uri);
  socket = new WebSocket(uri);
  // socket = new WebSocket('wss://localhost:6006/');

  // Connection opened
  socket.addEventListener('open', function(event) {
    console.log('connected');
    recordBtn.disabled = false;
    connectBtn.disabled = true;
    connectBtn.innerHTML = 'Connected!';
  });

  // Connection closed
  socket.addEventListener('close', function(event) {
    console.log('disconnected');
    recordBtn.disabled = true;
    connectBtn.disabled = false;
    connectBtn.innerHTML = 'Click me to connect!';
  });

  // Listen for messages
  socket.addEventListener('message', function(event) {
    let message = JSON.parse(event.data);
    if (message.segment in recognition_text) {
      recognition_text[message.segment] = message.text;
    } else {
      recognition_text.push(message.text);
    }
    let text_area = document.getElementById('results');
    text_area.value = getDisplayResult();
    text_area.scrollTop = text_area.scrollHeight;  // auto scroll
    console.log('Received message: ', event.data);
  });
}

window.onload = (event) => {
  console.log('page is fully loaded');
  console.log('protocol', window.location.protocol);
  console.log('port', window.location.port);
  if (window.location.protocol == 'https:') {
    document.getElementById('ws-protocol').textContent = 'wss://';
  }
  serverIpInput.value = window.location.hostname;
  serverPortInput.value = window.location.port;
};

const serverIpInput = document.getElementById('server-ip');
const serverPortInput = document.getElementById('server-port');

const connectBtn = document.getElementById('connect');
const recordBtn = document.getElementById('streaming_record');
const stopBtn = document.getElementById('streaming_stop');
const clearBtn = document.getElementById('clear');
const soundClips = document.getElementById('sound-clips');
const canvas = document.getElementById('canvas');
const mainSection = document.querySelector('.container');

stopBtn.disabled = true;
recordBtn.disabled = true;

let audioCtx;
const canvasCtx = canvas.getContext('2d');
let mediaStream;
let analyser;

let expectedSampleRate = 16000;
let recordSampleRate;  // the sampleRate of the microphone
let recorder = null;   // the microphone
let leftchannel = [];  // TODO: Use a single channel

let recordingLength = 0;  // number of samples so far

clearBtn.onclick = function() {
  document.getElementById('results').value = '';
  recognition_text = [];
};

connectBtn.onclick = function() {
  initWebSocket();
};

// copied/modified from https://mdn.github.io/web-dictaphone/
// and
// https://gist.github.com/meziantou/edb7217fddfbb70e899e
if (navigator.mediaDevices.getUserMedia) {
  console.log('getUserMedia supported.');

  // see https://w3c.github.io/mediacapture-main/#dom-mediadevices-getusermedia
  const constraints = {audio: true};

  let onSuccess = function(stream) {
    if (!audioCtx) {
      audioCtx = new AudioContext();
    }
    console.log(audioCtx);
    recordSampleRate = audioCtx.sampleRate;
    console.log('sample rate ' + recordSampleRate);

    // creates an audio node from the microphone incoming stream
    mediaStream = audioCtx.createMediaStreamSource(stream);
    console.log(mediaStream);

    // https://developer.mozilla.org/en-US/docs/Web/API/AudioContext/createScriptProcessor
    // bufferSize: the onaudioprocess event is called when the buffer is full
    var bufferSize = 2048;
    var numberOfInputChannels = 2;
    var numberOfOutputChannels = 2;
    if (audioCtx.createScriptProcessor) {
      recorder = audioCtx.createScriptProcessor(
          bufferSize, numberOfInputChannels, numberOfOutputChannels);
    } else {
      recorder = audioCtx.createJavaScriptNode(
          bufferSize, numberOfInputChannels, numberOfOutputChannels);
    }
    console.log(recorder);

    recorder.onaudioprocess = function(e) {
      let samples = new Float32Array(e.inputBuffer.getChannelData(0))
      samples = downsampleBuffer(samples, expectedSampleRate);

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

      socket.send(samples);

      leftchannel.push(buf);
      recordingLength += bufferSize;
    };

    visualize(stream);
    mediaStream.connect(analyser);

    recordBtn.onclick = function() {
      mediaStream.connect(recorder);
      mediaStream.connect(analyser);
      recorder.connect(audioCtx.destination);

      console.log('recorder started');
      recordBtn.style.background = 'red';

      stopBtn.disabled = false;
      recordBtn.disabled = true;
    };

    stopBtn.onclick = function() {
      console.log('recorder stopped');

      socket.send('Done');
      console.log('Sent Done');

      socket.close();

      // stopBtn recording
      recorder.disconnect(audioCtx.destination);
      mediaStream.disconnect(recorder);
      mediaStream.disconnect(analyser);

      recordBtn.style.background = '';
      recordBtn.style.color = '';
      // mediaRecorder.requestData();

      stopBtn.disabled = true;
      recordBtn.disabled = false;

      const clipName =
          prompt('Enter a name for your sound clip?', 'My unnamed clip');

      const clipContainer = document.createElement('article');
      const clipLabel = document.createElement('p');
      const audio = document.createElement('audio');
      const deleteButton = document.createElement('button');
      clipContainer.classList.add('clip');
      audio.setAttribute('controls', '');
      deleteButton.textContent = 'Delete';
      deleteButton.className = 'delete';

      if (clipName === null) {
        clipLabel.textContent = 'My unnamed clip';
      } else {
        clipLabel.textContent = clipName;
      }

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

function visualize(stream) {
  if (!audioCtx) {
    audioCtx = new AudioContext();
  }

  const source = audioCtx.createMediaStreamSource(stream);

  if (!analyser) {
    analyser = audioCtx.createAnalyser();
    analyser.fftSize = 2048;
  }
  const bufferLength = analyser.frequencyBinCount;
  const dataArray = new Uint8Array(bufferLength);

  // source.connect(analyser);
  // analyser.connect(audioCtx.destination);

  draw()

  function draw() {
    const WIDTH = canvas.width
    const HEIGHT = canvas.height;

    requestAnimationFrame(draw);

    analyser.getByteTimeDomainData(dataArray);

    canvasCtx.fillStyle = 'rgb(200, 200, 200)';
    canvasCtx.fillRect(0, 0, WIDTH, HEIGHT);

    canvasCtx.lineWidth = 2;
    canvasCtx.strokeStyle = 'rgb(0, 0, 0)';

    canvasCtx.beginPath();

    let sliceWidth = WIDTH * 1.0 / bufferLength;
    let x = 0;

    for (let i = 0; i < bufferLength; i++) {
      let v = dataArray[i] / 128.0;
      let y = v * HEIGHT / 2;

      if (i === 0) {
        canvasCtx.moveTo(x, y);
      } else {
        canvasCtx.lineTo(x, y);
      }

      x += sliceWidth;
    }

    canvasCtx.lineTo(canvas.width, canvas.height / 2);
    canvasCtx.stroke();
  }
}

window.onresize = function() {
  canvas.width = mainSection.offsetWidth;
};

window.onresize();

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
