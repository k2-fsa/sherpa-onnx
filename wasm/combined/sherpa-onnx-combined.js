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
    'sherpa-onnx-combined-core.js',
    'sherpa-onnx-combined-vad.js',
    'sherpa-onnx-combined-asr.js',
    'sherpa-onnx-combined-tts.js',
    'sherpa-onnx-combined-speaker.js',
    'sherpa-onnx-combined-enhancement.js',
    'sherpa-onnx-combined-kws.js'
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
    console.log("initialize() function called. Starting module loading process.");
    // Browser environment: load scripts
    if (typeof window !== 'undefined') {
      console.log("Browser environment detected. Proceeding to load modules sequentially.");
      // Load modules sequentially and handle completion/errors
      loadModulesSequentially()
        .catch(error => {
          console.error("Module loading failed:", error);
          // Ensure the callback is still called on failure, passing the error
          if (global.onSherpaOnnxReady) {
            console.log("Calling onSherpaOnnxReady with failure status due to error.");
            // Determine if any modules loaded successfully before the error
            let anyLoaded = Object.values(loadedModules).some(status => status === true);
            let missingModules = modulePaths.filter(path => !loadedModules[path]);
            global.onSherpaOnnxReady(anyLoaded && missingModules.length < modulePaths.length, error || missingModules);
          }
        });
    } else {
      console.log("Non-browser environment detected. Skipping module loading.");
    }
  };
  
  // Check if WASM module is already loaded
  if (typeof global.Module !== 'undefined' && typeof global.Module.onRuntimeInitialized !== 'undefined') {
    const originalOnRuntimeInitialized = global.Module.onRuntimeInitialized;
    
    global.Module.onRuntimeInitialized = function() {
      console.log("WASM module runtime initialized, checking for full initialization including HEAPF32...");
      
      if (originalOnRuntimeInitialized) {
        originalOnRuntimeInitialized();
      }
      
      // Wait for full initialization including HEAPF32
      let attempt = 0;
      const checkHeapInterval = setInterval(() => {
        attempt++;
        console.log(`Attempt ${attempt}: Checking if HEAPF32 is available...`);
        console.log(`global.Module.HEAPF32 exists: ${!!global.Module.HEAPF32}`);
        if (global.Module.HEAPF32) {
          console.log("HEAPF32 is available. Proceeding with JavaScript module initialization.");
          clearInterval(checkHeapInterval);
          initialize();
        } else if (attempt > 120) { // Wait up to 60 seconds (120 * 500ms)
          console.error("HEAPF32 not available after 60 seconds. Proceeding anyway with potential issues.");
          clearInterval(checkHeapInterval);
          initialize();
        }
      }, 500);
    };
  } else {
    // No WASM module yet, set up a listener
    global.onModuleReady = function() {
      console.log("WASM module ready, proceeding with module initialization");
      // Ensure HEAPF32 is available before proceeding
      if (global.Module && global.Module.HEAPF32) {
        console.log("HEAPF32 confirmed available via onModuleReady.");
        initialize();
      } else {
        console.error("onModuleReady called but HEAPF32 not available. Waiting for initialization.");
        let attempt = 0;
        const readyCheckInterval = setInterval(() => {
          attempt++;
          console.log(`Ready check attempt ${attempt}: Waiting for HEAPF32...`);
          if (global.Module && global.Module.HEAPF32) {
            console.log("HEAPF32 now available. Proceeding with initialization.");
            clearInterval(readyCheckInterval);
            initialize();
          } else if (attempt > 120) {
            console.error("HEAPF32 still not available after 60 seconds in onModuleReady. Proceeding with risk.");
            clearInterval(readyCheckInterval);
            initialize();
          }
        }, 500);
      }
    };
    
    // Since HEAPF32 availability was logged, check if it's already available and proceed
    if (typeof global.Module !== 'undefined' && global.Module.HEAPF32) {
      console.log("HEAPF32 already available. Triggering initialization immediately.");
      initialize();
    } else {
      console.log("Waiting for WASM module initialization or HEAPF32 availability before loading dependent scripts.");
      // Force initialization after a short timeout if no response
      console.log("Checking for HEAPF32 availability immediately for debugging.");
      if (typeof global.Module !== 'undefined') {
        console.log("Module exists. Current HEAPF32 status: ", !!global.Module.HEAPF32);
        if (global.Module.HEAPF32) {
          console.log("HEAPF32 detected. Proceeding with initialization NOW.");
          initialize();
        } else {
          console.log("No HEAPF32 yet. Waiting a very short period before forcing initialization.");
          setTimeout(() => {
            console.log("Immediate timeout reached. Forcing initialization regardless of HEAPF32 status to debug.");
            if (typeof global.Module !== 'undefined') {
              console.log("Module status at force: ", !!global.Module, "HEAPF32 status: ", !!global.Module.HEAPF32);
            } else {
              console.log("Module still not defined at force time.");
            }
            initialize();
          }, 1000); // Force after just 1 second for faster debugging
        }
      } else {
        console.log("Module not yet defined. Waiting for it to appear.");
        setTimeout(() => {
          console.log("Secondary timeout reached. Forcing initialization regardless of status.");
          initialize();
        }, 1000); // Force after just 1 second if Module isn't even defined
      }
    }
  }
})(typeof window !== 'undefined' ? window : global); 