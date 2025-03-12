

Module = {};
Module.onRuntimeInitialized = function() {
  console.log('Model files downloaded!');

  console.log('Initializing speech denoiser ......');
  tts = createOfflineSpeechDenoiser(Module)
  hint.innerText = 'Initialized! Please upload a wave file';
};
