const startBtn = document.getElementById('startBtn');
const hint = document.getElementById('hint');
const numClustersInput = document.getElementById('numClustersInputID');
const thresholdInput = document.getElementById('thresholdInputID');
const textArea = document.getElementById('text');

const fileSelectCtrl = document.getElementById('file');

let sd = null;
let float32Samples = null;

Module = {};
Module.onRuntimeInitialized = function() {
  console.log('Model files downloaded!');

  console.log('Initializing speaker diarization ......');
  sd = createOfflineSpeakerDiarization(Module)
  console.log('sampleRate', sd.sampleRate);

  hint.innerText =
      'Initialized! Please select a wave file and click the Start button.';

  fileSelectCtrl.disabled = false;
};

function onFileChange() {
  var files = document.getElementById('file').files;

  if (files.length == 0) {
    console.log('No file selected');
    float32Samples = null;
    startBtn.disabled = true;
    return;
  }
  textArea.value = '';

  console.log('files: ' + files);

  const file = files[0];
  console.log(file);
  console.log('file.name ' + file.name);
  console.log('file.type ' + file.type);
  console.log('file.size ' + file.size);

  let audioCtx = new AudioContext({sampleRate: sd.sampleRate});

  let reader = new FileReader();
  reader.onload = function() {
    console.log('reading file!');
    audioCtx.decodeAudioData(reader.result, decodedDone);
  };

  function decodedDone(decoded) {
    let typedArray = new Float32Array(decoded.length);
    float32Samples = decoded.getChannelData(0);

    startBtn.disabled = false;
  }

  reader.readAsArrayBuffer(file);
}

startBtn.onclick = function() {
  textArea.value = '';
  if (float32Samples == null) {
    alert('Empty audio samples!');

    startBtn.disabled = true;
    return;
  }

  let numClusters = numClustersInput.value;
  if (numClusters.trim().length == 0) {
    alert(
        'Please provide numClusters. Use -1 if you are not sure how many speakers are there');
    return;
  }

  if (!numClusters.match(/^\d+$/)) {
    alert(`number of clusters ${
        numClusters} is not an integer .\nPlease enter an integer`);
    return;
  }
  numClusters = parseInt(numClusters, 10);
  if (numClusters < -1) {
    alert(`Number of clusters should be >= -1`);
    return;
  }

  let threshold = 0.5;
  if (numClusters <= 0) {
    threshold = thresholdInput.value;
    if (threshold.trim().length == 0) {
      alert('Please provide a threshold.');
      return;
    }

    threshold = parseFloat(threshold);
    if (threshold < 0) {
      alert(`Pleaser enter a positive threshold`);
      return;
    }
  }

  let config = sd.config
  config.clustering = {numClusters: numClusters, threshold: threshold};
  sd.setConfig(config);
  let segments = sd.process(float32Samples);
  if (segments == null) {
    textArea.value = 'No speakers detected';
    return
  }

  let s = '';
  let sep = '';

  for (seg of segments) {
    // clang-format off
    s += sep + `${seg.start.toFixed(2)} -- ${seg.end.toFixed(2)} speaker_${seg.speaker}`
    // clang-format on
    sep = '\n';
  }
  textArea.value = s;
}
