
const fileInput = document.getElementById('fileInput');

let speech_denoiser = null;
const inAudioPlayback = document.getElementById('inAudioPlayback');
const outAudioPlayback = document.getElementById('outAudioPlayback');

Module = {};

// https://emscripten.org/docs/api_reference/module.html#Module.locateFile
Module.locateFile = function(path, scriptDirectory = '') {
  console.log(`path: ${path}, scriptDirectory: ${scriptDirectory}`);
  return scriptDirectory + path;
};

// https://emscripten.org/docs/api_reference/module.html#Module.locateFile
Module.setStatus = function(status) {
  console.log(`status ${status}`);
  const statusElement = document.getElementById('status');
  statusElement.textContent = status;
  if (status === '') {
    statusElement.style.display = 'none';
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
  console.log('Model files downloaded!');

  console.log('Initializing speech denoiser ......');
  speech_denoiser = createOfflineSpeechDenoiser(Module)
};

async function process(wave) {
  let denoised = speech_denoiser.run(wave.samples, wave.sampleRate);
  console.log(denoised);

  let int16Samples = new Int16Array(denoised.samples.length);
  for (var i = 0; i < denoised.samples.length; ++i) {
    let s = denoised.samples[i];
    if (s >= 1)
      s = 1;
    else if (s <= -1)
      s = -1;

    int16Samples[i] = s * 32767;
  }

  let blob = toWav(int16Samples, denoised.sampleRate);
  const objectUrl = URL.createObjectURL(blob);
  console.log(objectUrl);

  outAudioPlayback.src = objectUrl;
  outAudioPlayback.controls = true;
  outAudioPlayback.style.display = 'block';
}

fileInput.addEventListener('change', function(event) {
  if (!event.target.files || !event.target.files[0]) {
    console.log('No file selected.');
    return;
  }

  const file = event.target.files[0];
  console.log('Selected file:', file.name, file.type, file.size, 'bytes');
  const reader = new FileReader();
  reader.onload = function(ev) {
    console.log('FileReader onload called.');
    const arrayBuffer = ev.target.result;
    console.log('ArrayBuffer length:', arrayBuffer.byteLength);

    const uint8Array = new Uint8Array(arrayBuffer);
    const wave = readWaveFromBinaryData(uint8Array);


    var url = URL.createObjectURL(file);
    console.log(`url: ${url}`);
    inAudioPlayback.src = url;
    inAudioPlayback.style.display = 'block';

    process(wave)
    console.log('process done')
  };
  reader.onerror = function(err) {
    console.error('FileReader error:', err);
  };
  console.log('Starting FileReader.readAsArrayBuffer...');
  reader.readAsArrayBuffer(file);
});

// this function is copied/modified from
// https://gist.github.com/meziantou/edb7217fddfbb70e899e
function toWav(samples, sampleRate) {
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
