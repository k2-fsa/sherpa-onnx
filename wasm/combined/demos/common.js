// Set up initialization callback
window.onSherpaOnnxReady = function(success, error) {
  if (success) {
    console.log("All SherpaOnnx modules loaded successfully");
    initializeUI(); // This function would be defined in each individual demo file
  } else {
    console.error("Some SherpaOnnx modules failed to load:", error);
    document.getElementById('status').textContent = 
      "Error loading some modules. Some features may not work correctly.";
    document.getElementById('status').style.backgroundColor = "#ffcccc";
    
    // Still try to initialize the UI with available modules
    initializeUI();
  }
};

// Old-style module initialization for backward compatibility
window.onModuleReady = function() {
  console.log("WASM module ready - waiting for all JS modules to load");
};

// Shared audio context and microphone access
let audioContext;
let mediaStream;

function setupAudioContext() {
  if (!audioContext) {
    audioContext = new (window.AudioContext || window.webkitAudioContext)({sampleRate: 16000});
  }
  return audioContext;
}

async function getMicrophoneInput() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({audio: true});
    const context = setupAudioContext();
    mediaStream = context.createMediaStreamSource(stream);
    return stream;
  } catch (error) {
    console.error('Error accessing microphone:', error);
    throw error;
  }
}

// Create unload button
function createUnloadButton(container, modelType, resource, statusElem) {
  const button = document.createElement('button');
  button.textContent = `Unload ${modelType} Model`;
  button.classList.add('unload-button');
  
  button.addEventListener('click', function() {
    if (resource) {
      // Free the resource
      resource.free();
      
      // Call the appropriate cleanup method
      if (modelType === 'ASR') {
        SherpaOnnx.cleanupASR();
      } else if (modelType === 'TTS') {
        SherpaOnnx.cleanupTTS();
      } else if (modelType === 'VAD') {
        SherpaOnnx.cleanupVAD();
      } else if (modelType === 'KWS') {
        SherpaOnnx.cleanupKWS();
      }
      
      // Update UI
      button.disabled = true;
      if (statusElem) {
        statusElem.textContent = `Status: ${modelType} model unloaded`;
      }
      
      console.log(`${modelType} model unloaded successfully`);
    }
  });
  
  container.appendChild(button);
  return button;
}
