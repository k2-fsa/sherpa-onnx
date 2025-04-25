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

// Validate WASM filesystem assets
function validateAssets(targetElement, moduleTypes = ['vad', 'tts', 'asr', 'kws']) {
  if (!window.SherpaOnnx || !window.SherpaOnnx.FileSystem) {
    targetElement.innerHTML = '<div class="error">SherpaOnnx FileSystem not available</div>';
    return false;
  }

  const fs = window.SherpaOnnx.FileSystem;
  const container = document.createElement('div');
  container.className = 'filesystem-inspector';
  
  // Check root directories
  const rootSection = document.createElement('div');
  rootSection.className = 'fs-section';
  rootSection.innerHTML = '<h3>Root Directory</h3>';
  
  try {
    const rootFiles = fs.listFiles('/');
    
    if (rootFiles.length === 0) {
      rootSection.innerHTML += '<div class="warning">No files found in root directory</div>';
    } else {
      const rootList = document.createElement('ul');
      rootFiles.forEach(file => {
        const item = document.createElement('li');
        item.textContent = file;
        rootList.appendChild(item);
      });
      rootSection.appendChild(rootList);
    }
  } catch (e) {
    rootSection.innerHTML += `<div class="error">Error listing root directory: ${e.message}</div>`;
  }
  
  container.appendChild(rootSection);
  
  // Check each module type
  moduleTypes.forEach(moduleType => {
    const section = document.createElement('div');
    section.className = 'fs-section';
    section.innerHTML = `<h3>${moduleType.toUpperCase()} Assets</h3>`;
    
    // Check various possible asset paths - updated to include sherpa_assets
    const assetPaths = [
      `/sherpa_assets/${moduleType}`,  // Added - This is the correct path per CMakeLists.txt
      `/assets/${moduleType}`,
      `/assets/${moduleType}/models`,
      `/preloaded/${moduleType}`
    ];
    
    let assetsFound = false;
    
    assetPaths.forEach(assetPath => {
      if (fs.fileExists(assetPath)) {
        assetsFound = true;
        const files = fs.listFiles(assetPath);
        
        const pathDiv = document.createElement('div');
        pathDiv.className = 'path-info';
        pathDiv.innerHTML = `<strong>${assetPath}:</strong>`;
        
        if (files.length === 0) {
          pathDiv.innerHTML += ' <span class="warning">Directory exists but is empty</span>';
        } else {
          const fileList = document.createElement('ul');
          files.forEach(file => {
            const item = document.createElement('li');
            item.textContent = file;
            fileList.appendChild(item);
          });
          pathDiv.appendChild(fileList);
        }
        
        section.appendChild(pathDiv);
      }
    });
    
    if (!assetsFound) {
      section.innerHTML += `<div class="warning">No ${moduleType} asset directories found</div>`;
    }
    
    container.appendChild(section);
  });
  
  // Also check if sherpa_assets directory exists
  try {
    if (fs.fileExists('/sherpa_assets')) {
      const sherpaSection = document.createElement('div');
      sherpaSection.className = 'fs-section';
      sherpaSection.innerHTML = '<h3>Sherpa Assets Directory</h3>';
      
      const sherpaFiles = fs.listFiles('/sherpa_assets');
      if (sherpaFiles.length === 0) {
        sherpaSection.innerHTML += '<div class="warning">Directory exists but is empty</div>';
      } else {
        const pathDiv = document.createElement('div');
        pathDiv.className = 'path-info';
        pathDiv.innerHTML = '<strong>/sherpa_assets:</strong>';
        
        const fileList = document.createElement('ul');
        sherpaFiles.forEach(file => {
          const item = document.createElement('li');
          item.textContent = file;
          
          // Recursively show contents for each subdirectory
          if (fs.fileExists(`/sherpa_assets/${file}`)) {
            try {
              const subFiles = fs.listFiles(`/sherpa_assets/${file}`);
              if (subFiles.length > 0) {
                const subList = document.createElement('ul');
                subFiles.forEach(subFile => {
                  const subItem = document.createElement('li');
                  subItem.textContent = subFile;
                  subList.appendChild(subItem);
                });
                item.appendChild(subList);
              }
            } catch (e) {
              // Ignore errors for subdir listing
            }
          }
          
          fileList.appendChild(item);
        });
        
        pathDiv.appendChild(fileList);
        sherpaSection.appendChild(pathDiv);
      }
      
      container.appendChild(sherpaSection);
    }
  } catch (e) {
    // Ignore errors if sherpa_assets doesn't exist
  }
  
  // Clear and update target element
  targetElement.innerHTML = '';
  targetElement.appendChild(container);
  
  // Add some basic styling
  const style = document.createElement('style');
  style.textContent = `
    .filesystem-inspector {
      font-family: monospace;
      background-color: #f5f5f5;
      border: 1px solid #ddd;
      border-radius: 4px;
      padding: 10px;
      margin: 10px 0;
      max-height: 400px;
      overflow-y: auto;
    }
    .fs-section {
      margin-bottom: 15px;
    }
    .fs-section h3 {
      margin: 0 0 5px 0;
      font-size: 1em;
      color: #333;
    }
    .path-info {
      margin: 5px 0;
      padding-left: 10px;
    }
    .error {
      color: #cc0000;
      font-weight: bold;
    }
    .warning {
      color: #cc7700;
    }
    ul {
      margin: 5px 0;
      padding-left: 20px;
    }
  `;
  targetElement.appendChild(style);
  
  return true;
}

// Create inspect assets button
function createInspectAssetsButton(container, targetElement) {
  const button = document.createElement('button');
  button.textContent = 'Inspect Filesystem Assets';
  button.classList.add('inspect-button');
  
  button.addEventListener('click', function() {
    validateAssets(targetElement);
  });
  
  container.appendChild(button);
  return button;
}
