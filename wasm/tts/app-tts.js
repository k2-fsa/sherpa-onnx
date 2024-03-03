const generateBtn = document.getElementById('generateBtn');
const hint = document.getElementById('hint');
const speakerIdLabel = document.getElementById('speakerIdLabel');
const speakerIdInput = document.getElementById('speakerId');
const speedInput = document.getElementById('speed');
const speedValue = document.getElementById('speedValue');
const textArea = document.getElementById('text');
const soundClips = document.getElementById('sound-clips');

speedValue.innerHTML = speedInput.value;

let index = 0;


let tts = null;

let audioCtx = null;


Module = {};
Module.onRuntimeInitialized = function() {
  console.log('Model files downloaded!');

  console.log('Initializing tts ......');
  tts = createOfflineTts(Module)
  if (tts.numSpeakers > 1) {
    speakerIdLabel.innerHTML = `Speaker ID (0 - ${tts.numSpeakers - 1}):`;
  }

  hint.innerText =
      'Initialized! Please enter text and click the Generate button.';



  generateBtn.disabled = false;
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
        tts.numSpeakers - 1}`);
    return;
  }
  speakerId = parseInt(speakerId, 10);
  if (speakerId > tts.numSpeakers - 1) {
    alert(`Pleaser enter a number between 0 and ${tts.numSpeakers - 1}`);
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

  let audio =
      tts.generate({text: text, sid: speakerId, speed: speedInput.value});

  console.log(audio.samples.length, audio.sampleRate);

  if (!audioCtx) {
    audioCtx = new AudioContext({sampleRate: tts.sampleRate});
  }

  const buffer = audioCtx.createBuffer(1, audio.samples.length, tts.sampleRate);

  const ptr = buffer.getChannelData(0);
  for (let i = 0; i < audio.samples.length; i++) {
    ptr[i] = audio.samples[i];
  }
  const source = audioCtx.createBufferSource();
  source.buffer = buffer;
  source.connect(audioCtx.destination);
  source.start();

  createAudioTag(audio);
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
