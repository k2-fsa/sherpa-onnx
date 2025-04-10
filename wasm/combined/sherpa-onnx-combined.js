/**
 * sherpa-onnx-combined.js
 * 
 * Loader for all Sherpa-ONNX modules
 */

(function(global) {
  // Auto-detect script path to handle loading from different directories
  function getScriptPath() {
    // For browser environments
    if (typeof document !== 'undefined') {
      const scripts = document.getElementsByTagName('script');
      for (let i = 0; i < scripts.length; i++) {
        const src = scripts[i].src;
        if (src.indexOf('sherpa-onnx-combined.js') !== -1) {
          // Return the directory path of the script
          return src.substring(0, src.lastIndexOf('/') + 1);
        }
      }
    }
    // Default path if we can't detect
    return '';
  }
  
  // Get the base path where all JS modules are located
  const basePath = getScriptPath();
  console.log("Detected script base path:", basePath);

  // Define module paths relative to the base path
  const defaultModules = [
    'sherpa-onnx-core.js',
    'sherpa-onnx-vad.js',
    'sherpa-onnx-asr.js',
    'sherpa-onnx-tts.js',
    'sherpa-onnx-speaker.js',
    'sherpa-onnx-enhancement.js',
    'sherpa-onnx-kws.js'
  ];
  
  // Use custom module paths if provided, otherwise use defaults with base path
  let modulePaths;
  if (typeof window !== 'undefined' && window.sherpaOnnxModulePaths) {
    console.log("Using custom module paths from window.sherpaOnnxModulePaths");
    modulePaths = window.sherpaOnnxModulePaths;
  } else if (global.sherpaOnnxModulePaths) {
    console.log("Using custom module paths from global.sherpaOnnxModulePaths");
    modulePaths = global.sherpaOnnxModulePaths;
  } else {
    // Apply base path to each module
    modulePaths = defaultModules.map(module => basePath + module);
    console.log("Using default module paths with detected base path:", modulePaths);
  }
  
  // Keep track of loaded modules
  let loadedModules = {};
  let modulesLoading = false;
  
  // Keep track of active resources to clean up
  let activeResources = {
    asr: [],
    tts: [],
    vad: [],
    speaker: [],
    enhancement: [],
    kws: []
  };
  
  // Async loader for scripts
  const loadScript = function(url) {
    return new Promise((resolve, reject) => {
      const script = document.createElement('script');
      script.src = url;
      script.async = true;
      
      script.onload = () => {
        console.log(`Module ${url} loaded successfully`);
        loadedModules[url] = true;
        resolve();
      };
      
      script.onerror = (e) => {
        console.error(`Failed to load script: ${url}`, e);
        loadedModules[url] = false;
        // Continue loading other modules even if one fails
        resolve();
      };
      
      document.head.appendChild(script);
    });
  };
  
  // Check if core module is available
  const ensureCoreModule = function() {
    if (!global.SherpaOnnx) {
      console.error("SherpaOnnx core module not loaded! Other modules will not function properly.");
      return false;
    }
    return true;
  };
  
  // Load modules in sequence to ensure proper initialization
  const loadModulesSequentially = async function() {
    if (modulesLoading) return;
    
    modulesLoading = true;
    
    try {
      // Load core module first since other modules depend on it
      console.log("Loading SherpaOnnx core module from: " + modulePaths[0]);
      await loadScript(modulePaths[0]); // Use the first module from the paths array
      
      if (!ensureCoreModule()) {
        throw new Error("Failed to load core module");
      }
      
      // Load the rest of the modules sequentially
      for (let i = 1; i < modulePaths.length; i++) {
        console.log(`Loading module ${i+1}/${modulePaths.length}: ${modulePaths[i]}`);
        await loadScript(modulePaths[i]);
      }
      
      // Check if all critical modules are loaded
      let allLoaded = true;
      let missingModules = [];
      
      for (const module of modulePaths) {
        if (!loadedModules[module]) {
          allLoaded = false;
          missingModules.push(module);
        }
      }
      
      if (!allLoaded) {
        console.warn(`Not all modules loaded successfully. Missing: ${missingModules.join(', ')}`);
      } else {
        console.log("All SherpaOnnx modules loaded successfully");
      }
      
      // Add resource tracking and cleanup methods after modules are loaded
      if (global.SherpaOnnx) {
        // Add resource tracking methods
        global.SherpaOnnx.trackResource = function(type, resource) {
          if (activeResources[type]) {
            activeResources[type].push(resource);
          }
          return resource;
        };
        
        // Add cleanup methods
        global.SherpaOnnx.cleanup = function(type) {
          if (!type) {
            // Clean up all resource types if no specific type is provided
            Object.keys(activeResources).forEach(t => this.cleanup(t));
            return;
          }
          
          if (activeResources[type]) {
            const resources = activeResources[type];
            console.log(`Cleaning up ${resources.length} ${type} resources`);
            
            for (let i = resources.length - 1; i >= 0; i--) {
              try {
                if (resources[i] && typeof resources[i].free === 'function') {
                  resources[i].free();
                }
                resources.splice(i, 1);
              } catch (e) {
                console.error(`Error cleaning up ${type} resource:`, e);
              }
            }
          }
        };
        
        // Add convenience methods for each resource type
        global.SherpaOnnx.cleanupASR = function() { this.cleanup('asr'); };
        global.SherpaOnnx.cleanupTTS = function() { this.cleanup('tts'); };
        global.SherpaOnnx.cleanupVAD = function() { this.cleanup('vad'); };
        global.SherpaOnnx.cleanupSpeaker = function() { this.cleanup('speaker'); };
        global.SherpaOnnx.cleanupEnhancement = function() { this.cleanup('enhancement'); };
        global.SherpaOnnx.cleanupKWS = function() { this.cleanup('kws'); };
      }
      
      // Call ready callback if defined
      if (global.onSherpaOnnxReady) {
        console.log("Calling onSherpaOnnxReady callback");
        global.onSherpaOnnxReady(allLoaded, missingModules);
      }
    } catch (error) {
      console.error("Error during module loading:", error);
      
      if (global.onSherpaOnnxReady) {
        global.onSherpaOnnxReady(false, error);
      }
    } finally {
      modulesLoading = false;
    }
  };
  
  // Main initialization function
  const initialize = function() {
    // Browser environment: load scripts
    if (typeof window !== 'undefined') {
      // Set up a backup timeout to ensure callback is called even if loading fails
      const timeoutPromise = new Promise((resolve) => {
        setTimeout(() => {
          console.warn("Module loading timeout reached - some modules may not have loaded correctly");
          resolve();
        }, 30000); // 30 second timeout
      });
      
      // Load modules with timeout protection
      Promise.race([loadModulesSequentially(), timeoutPromise])
        .catch(error => {
          console.error("Module loading failed:", error);
          
          if (global.onSherpaOnnxReady) {
            global.onSherpaOnnxReady(false, error);
          }
        });
    }
  };
  
  // Check if WASM module is already loaded
  if (typeof global.Module !== 'undefined' && typeof global.Module.onRuntimeInitialized !== 'undefined') {
    const originalOnRuntimeInitialized = global.Module.onRuntimeInitialized;
    
    global.Module.onRuntimeInitialized = function() {
      console.log("WASM module runtime initialized, now loading JavaScript modules");
      
      if (originalOnRuntimeInitialized) {
        originalOnRuntimeInitialized();
      }
      
      initialize();
    };
  } else {
    // No WASM module yet, set up a listener
    global.onModuleReady = function() {
      console.log("WASM module ready, proceeding with module initialization");
      initialize();
    };
    
    // Also start loading anyway in case the event was missed
    setTimeout(initialize, 1000);
  }
})(typeof window !== 'undefined' ? window : global); 