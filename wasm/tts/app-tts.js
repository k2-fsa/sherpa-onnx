const generateBtn = document.getElementById('generateBtn');
const speakerIdLabel = document.getElementById('speakerIdLabel');
const speakerIdInput = document.getElementById('speakerId');
const speedInput = document.getElementById('speed');
const speedValue = document.getElementById('speedValue');
const textArea = document.getElementById('text');
const soundClips = document.getElementById('sound-clips');

speedValue.innerHTML = speedInput.value;

let index = 0;

let audioCtx = null;
const worker = new Worker("/sherpa-onnx-tts.worker.js");
let ttsInstanceInfo = {
  numSpeakers: 0,
  isReady: false,
};
worker.onmessage = (e) => {
  if (e.data.type === "sherpa-onnx-tts-progress") {
    Module.setStatus(e.data.status);
  }
  if (e.data.type === "sherpa-onnx-tts-ready") {
    ttsInstanceInfo.numSpeakers = e.data.numSpeakers;
    ttsInstanceInfo.isReady = true;
    generateBtn.disabled = false;
    speakerIdLabel.innerHTML = `Speaker ID (0 - ${e.data.numSpeakers - 1}):`;
    return;
  }
  if (e.data.type === "sherpa-onnx-tts-result") {
    let audio = e.data;

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

Module = {};

// https://emscripten.org/docs/api_reference/module.html#Module.locateFile
Module.setStatus = function(status) {
  console.log(`status ${status}`);
  const statusElement = document.getElementById('status');
  if (status == 'Running...') {
    status = 'Model downloaded. Initializing text to speech model...'
  }

  const downloadMatch = status.match(/Downloading data... \((\d+)\/(\d+)\)/);
  if (downloadMatch) {
    const downloaded = BigInt(downloadMatch[1]);
    const total = BigInt(downloadMatch[2]);
    const percent =
        total === 0 ? 0.00 : Number((downloaded * 10000n) / total) / 100;
    status = `Downloading data... ${percent.toFixed(2)}% (${downloadMatch[1]}/${
        downloadMatch[2]})`;
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
speedInput.oninput = function() {
  speedValue.innerHTML = this.value;
};

generateBtn.onclick = function() {
  let speakerId = speakerIdInput.value;
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

  let text = textArea.value.trim();
  if (text.length == 0) {
    alert('Please input a non-blank text');
    return;
  }

  console.log('speakerId', speakerId);
  console.log('speed', speedInput.value);
  console.log('text', text);
  worker.postMessage({
    text,
    sid: speakerId,
    speed: speedInput.value,
    type: "generate",
  });
};

function createAudioTag(generateAudio) {
  const blob = toWav(generateAudio.samples, generateAudio.sampleRate);

  const text = textArea.value.trim().substring(0, 100);
  const clipName = `${index} ${text} ...`;
  index += 1;

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
