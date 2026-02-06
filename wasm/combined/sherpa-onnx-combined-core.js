/**
 * sherpa-onnx-core.js
 * 
 * Core functionality for the SherpaOnnx WASM modules
 */

(function(global) {
  // Create the SherpaOnnx namespace if it doesn't exist
  global.SherpaOnnx = global.SherpaOnnx || {};
  
  // Create main namespace
  const SherpaOnnx = {};
  
  // Check if Module already exists and extend it
  if (typeof window.Module !== 'undefined') {
    console.log('Module already defined at script load time. Checking initialization status...');
    console.log('Module properties at load:', Object.keys(window.Module).slice(0, 10), '... (first 10 shown)');
    console.log('Module.onRuntimeInitialized exists:', !!window.Module.onRuntimeInitialized);
    console.log('Module.calledRun status at load:', !!window.Module.calledRun);
    // Immediate attempt to initialize HEAPF32 at load time
    if (!window.Module.HEAPF32) {
      try {
        if (window.Module.HEAP8) {
          window.Module.HEAPF32 = new Float32Array(window.Module.HEAP8.buffer);
          console.log('Successfully initialized HEAPF32 dynamically from HEAP8 at load time in core module.');
        } else if (window.Module.asm && window.Module.asm.memory) {
          window.Module.HEAPF32 = new Float32Array(window.Module.asm.memory.buffer);
          console.log('Successfully initialized HEAPF32 directly from WebAssembly memory at load time in core module.');
        } else if (window.Module.memory) {
          window.Module.HEAPF32 = new Float32Array(window.Module.memory.buffer);
          console.log('Successfully initialized HEAPF32 from Module.memory at load time in core module.');
        } else if (window.Module._memory) {
          window.Module.HEAPF32 = new Float32Array(window.Module._memory.buffer);
          console.log('Successfully initialized HEAPF32 from Module._memory at load time in core module.');
        } else if (typeof WebAssembly !== 'undefined' && WebAssembly.Memory && window.Module.asm) {
          for (const prop in window.Module.asm) {
            if (window.Module.asm[prop] instanceof WebAssembly.Memory) {
              window.Module.HEAPF32 = new Float32Array(window.Module.asm[prop].buffer);
              console.log(`Successfully initialized HEAPF32 from WebAssembly.Memory found in asm.${prop} at load time in core module.`);
              break;
            }
          }
          if (!window.Module.HEAPF32) {
            console.warn('No WebAssembly.Memory found in asm properties at load time in core module.');
          }
        } else {
          console.warn('No standard method found to initialize HEAPF32 at load time in core module.');
          // Simplified deeper inspection of window.Module for any memory buffer
          console.log('Inspecting window.Module for potential memory buffers...');
          let foundBuffer = false;
          for (const prop in window.Module) {
            try {
              if (window.Module[prop] && typeof window.Module[prop] === 'object') {
                if (window.Module[prop] instanceof ArrayBuffer) {
                  window.Module.HEAPF32 = new Float32Array(window.Module[prop]);
                  console.log(`Initialized HEAPF32 from ArrayBuffer in Module.${prop} at load time.`);
                  foundBuffer = true;
                  break;
                } else if (window.Module[prop].buffer && window.Module[prop].buffer instanceof ArrayBuffer) {
                  window.Module.HEAPF32 = new Float32Array(window.Module[prop].buffer);
                  console.log(`Initialized HEAPF32 from buffer in Module.${prop}.buffer at load time.`);
                  foundBuffer = true;
                  break;
                }
              }
            } catch (e) {
              console.error(`Error inspecting Module.${prop} at load time:`, e.message);
            }
          }
          if (!foundBuffer) {
            console.log('No suitable memory buffer found in deep inspection at load time.');
          }
        }
      } catch (e) {
        console.error('Failed to initialize HEAPF32 dynamically at load time in core module:', e.message);
      }
      console.log(`Post-workaround at load time - HEAPF32 exists: ${!!window.Module.HEAPF32}`);
    }
    const originalOnRuntimeInitialized = window.Module.onRuntimeInitialized;
    window.Module.onRuntimeInitialized = function() {
      console.log('onRuntimeInitialized triggered. SherpaOnnx Core module initialized.');
      console.log('Module.calledRun status when onRuntimeInitialized triggered:', !!window.Module.calledRun);
      console.log('Checking for HEAPF32 availability after initialization:', !!window.Module.HEAPF32);
      global.SherpaOnnx.isReady = true; // Custom readiness flag
      console.log('SherpaOnnx readiness flag set to true');
      if (originalOnRuntimeInitialized) {
        console.log('Calling original onRuntimeInitialized callback.');
        originalOnRuntimeInitialized();
      }
      if (global.onModuleReady) {
        console.log('Calling global.onModuleReady callback.');
        global.onModuleReady();
      }
    };
    console.log('onRuntimeInitialized hook set. Waiting for initialization...');
    // Additional check if calledRun is already true but onRuntimeInitialized hasn't fired
    if (window.Module.calledRun && !global.SherpaOnnx.isReady) {
      console.warn('Module.calledRun is true but onRuntimeInitialized has not fired. Forcing readiness check.');
      // Start a continuous check for HEAPF32 availability
      let heapCheckAttempts = 0;
      const heapCheckInterval = setInterval(() => {
        heapCheckAttempts++;
        console.log(`HEAPF32 check attempt ${heapCheckAttempts}: HEAPF32 exists: ${!!window.Module.HEAPF32}`);
        if (!window.Module.HEAPF32) {
          try {
            if (window.Module.HEAP8) {
              window.Module.HEAPF32 = new Float32Array(window.Module.HEAP8.buffer);
              console.log('Initialized HEAPF32 from HEAP8 during continuous check.');
            } else if (window.Module.asm && window.Module.asm.memory) {
              window.Module.HEAPF32 = new Float32Array(window.Module.asm.memory.buffer);
              console.log('Initialized HEAPF32 from WebAssembly memory during continuous check.');
            } else if (window.Module.memory) {
              window.Module.HEAPF32 = new Float32Array(window.Module.memory.buffer);
              console.log('Initialized HEAPF32 from Module.memory during continuous check.');
            } else if (window.Module._memory) {
              window.Module.HEAPF32 = new Float32Array(window.Module._memory.buffer);
              console.log('Initialized HEAPF32 from Module._memory during continuous check.');
            } else if (typeof WebAssembly !== 'undefined' && WebAssembly.Memory && window.Module.asm) {
              for (const prop in window.Module.asm) {
                if (window.Module.asm[prop] instanceof WebAssembly.Memory) {
                  window.Module.HEAPF32 = new Float32Array(window.Module.asm[prop].buffer);
                  console.log(`Initialized HEAPF32 from WebAssembly.Memory in asm.${prop} during continuous check.`);
                  break;
                }
              }
              if (!window.Module.HEAPF32) {
                console.warn('No WebAssembly.Memory found in asm properties during continuous check.');
              }
            } else {
              console.warn('No standard method found to initialize HEAPF32 during continuous check.');
              // Simplified deeper inspection during continuous check
              console.log(`Check ${heapCheckAttempts}: Inspecting window.Module for memory buffers...`);
              let foundBuffer = false;
              for (const prop in window.Module) {
                try {
                  if (window.Module[prop] && typeof window.Module[prop] === 'object') {
                    if (window.Module[prop] instanceof ArrayBuffer) {
                      window.Module.HEAPF32 = new Float32Array(window.Module[prop]);
                      console.log(`Initialized HEAPF32 from ArrayBuffer in Module.${prop} during check ${heapCheckAttempts}.`);
                      foundBuffer = true;
                      break;
                    } else if (window.Module[prop].buffer && window.Module[prop].buffer instanceof ArrayBuffer) {
                      window.Module.HEAPF32 = new Float32Array(window.Module[prop].buffer);
                      console.log(`Initialized HEAPF32 from buffer in Module.${prop}.buffer during check ${heapCheckAttempts}.`);
                      foundBuffer = true;
                      break;
                    }
                  }
                } catch (e) {
                  console.error(`Error inspecting Module.${prop} during check ${heapCheckAttempts}:`, e.message);
                }
              }
              if (!foundBuffer) {
                console.log(`Check ${heapCheckAttempts}: No suitable memory buffer found in deep inspection.`);
              }
            }
          } catch (e) {
            console.error('Failed to initialize HEAPF32 during continuous check:', e.message);
          }
        }
        if (window.Module.HEAPF32 || heapCheckAttempts >= 10) {
          clearInterval(heapCheckInterval);
          console.log(`Stopping HEAPF32 checks after ${heapCheckAttempts} attempts. Final status - HEAPF32 exists: ${!!window.Module.HEAPF32}`);
          if (!window.Module.HEAPF32) {
            console.error('HEAPF32 initialization failed after maximum attempts. Proceeding anyway to unblock UI.');
          }
          if (!global.SherpaOnnx.isReady) {
            global.SherpaOnnx.isReady = true;
            console.log('SherpaOnnx readiness flag manually set to true after HEAPF32 check.');
            if (global.onModuleReady) {
              console.log('Calling global.onModuleReady callback after HEAPF32 check.');
              global.onModuleReady();
            }
          }
        }
      }, 500); // Check every 500ms up to 10 attempts (5 seconds)
      setTimeout(() => {
        if (!global.SherpaOnnx.isReady) {
          console.error('onRuntimeInitialized still not triggered after extended delay. Manually setting readiness flag.');
          global.SherpaOnnx.isReady = true;
          console.log('SherpaOnnx readiness flag manually set to true due to timeout.');
          if (!window.Module.HEAPF32) {
            console.log('HEAPF32 still not available after timeout. Final attempt to initialize.');
            try {
              if (window.Module.HEAP8) {
                window.Module.HEAPF32 = new Float32Array(window.Module.HEAP8.buffer);
                console.log('Initialized HEAPF32 from HEAP8 during final timeout check.');
              } else if (window.Module.asm && window.Module.asm.memory) {
                window.Module.HEAPF32 = new Float32Array(window.Module.asm.memory.buffer);
                console.log('Initialized HEAPF32 from WebAssembly memory during final timeout check.');
              }
            } catch (e) {
              console.error('Final attempt to initialize HEAPF32 failed:', e.message);
            }
            console.log(`Final status after timeout - HEAPF32 exists: ${!!window.Module.HEAPF32}`);
          }
          if (global.onModuleReady) {
            console.log('Calling global.onModuleReady callback due to forced readiness.');
            global.onModuleReady();
          }
        }
      }, 10000); // Wait 10 seconds before forcing readiness
    }
  } else {
    console.log('Module not defined at script load time. Setting up property trap...');
    console.log("Waiting for Module to be defined...");
    Object.defineProperty(global, 'Module', {
      set: function(mod) {
        console.log('Module being set. Capturing initialization...');
        console.log('Module properties at set:', Object.keys(mod).slice(0, 10), '... (first 10 shown)');
        this._Module = mod;
        console.log("Module defined, waiting for runtime initialization");
        const originalOnRuntimeInitialized = mod.onRuntimeInitialized;
        mod.onRuntimeInitialized = function() {
          console.log("onRuntimeInitialized triggered from setter. SherpaOnnx Core module initialized.");
          global.SherpaOnnx.isReady = true; // Custom readiness flag
          console.log("SherpaOnnx readiness flag set to true from setter");
          if (originalOnRuntimeInitialized) {
            console.log('Calling original onRuntimeInitialized callback from setter.');
            originalOnRuntimeInitialized();
          }
          if (global.onModuleReady) {
            console.log('Calling global.onModuleReady callback from setter.');
            global.onModuleReady();
          }
        };
        console.log('onRuntimeInitialized hook set in setter. Waiting for initialization...');
      },
      get: function() {
        return this._Module;
      }
    });
  }
  
  // Configuration for SherpaOnnx
  SherpaOnnx.Config = {
    // Paths to preloaded assets
    assetPaths: {
      vad: '/sherpa_assets/vad',
      tts: '/sherpa_assets/tts',
      asr: '/sherpa_assets/asr',
      kws: '/sherpa_assets/kws',
      speakers: '/sherpa_assets/speakers',
      enhancement: '/sherpa_assets/enhancement'
    },

    // Allow users to override the location of the data file
    setDataFileLocation: function(location) {
      if (global.Module) {
        const originalLocateFile = global.Module.locateFile;
        global.Module.locateFile = function(path) {
          if (path.endsWith('.data')) {
            return location;
          }
          return typeof originalLocateFile === 'function' 
            ? originalLocateFile(path) 
            : path;
        };
      }
    }
  };
  
  // Common utilities for memory management and shared functionality
  SherpaOnnx.Utils = {
    /**
     * Free configuration memory allocated in WASM
     * @param {Object} config - Configuration object with allocated memory
     * @param {Object} Module - WebAssembly module
     */
    freeConfig: function(config, Module) {
      if (!config) return;
      
      if ('buffer' in config) {
        Module._free(config.buffer);
      }
  
      if ('sileroVad' in config) {
        this.freeConfig(config.sileroVad, Module);
      }
  
      if (config.ptr) {
        Module._free(config.ptr);
      }
    },
    
    /**
     * Copy string to WASM heap and return pointer
     * @param {string} str - String to allocate
     * @param {Object} Module - WebAssembly module
     * @returns {Object} Object with pointer and length
     */
    allocateString: function(str, Module) {
      if (!str) str = '';
      const strLen = Module.lengthBytesUTF8(str) + 1;
      const strPtr = Module._malloc(strLen);
      Module.stringToUTF8(str, strPtr, strLen);
      return { ptr: strPtr, len: strLen };
    }
  };
  
  // File system utilities for model loading
  SherpaOnnx.FileSystem = {
    /**
     * Check if a file exists in the filesystem
     * @param {string} path - Path to check
     * @returns {boolean} - Whether the file exists
     */
    fileExists: function(path) {
      try {
        global.Module.FS.lookupPath(path);
        return true;
      } catch (e) {
        return false;
      }
    },
    
    /**
     * Get a valid asset path for a given module type and filename
     * @param {string} moduleType - Type of module (vad, tts, kws, asr)
     * @param {string} filename - Name of the file to look for
     * @returns {string} - The first valid path where the asset exists
     */
    getAssetPath: function(moduleType, filename) {
      // Check in the preloaded assets directory structure
      const paths = [
        `/assets/${moduleType}/${filename}`,
        `/assets/${moduleType}/models/${filename}`,
        `/preloaded/${moduleType}/${filename}`
      ];
      
      // Return first path that exists
      for (const path of paths) {
        if (this.fileExists(path)) {
          return path;
        }
      }
      
      // Default fallback
      return `/assets/${moduleType}/${filename}`;
    },
    
    /**
     * List files in a directory in the filesystem
     * @param {string} dirPath - Directory path
     * @returns {Array<string>} List of files
     */
    listFiles: function(dirPath) {
      try {
        if (!global.Module || !global.Module.FS) return [];
        return global.Module.FS.readdir(dirPath).filter(
          name => name !== '.' && name !== '..'
        );
      } catch (e) {
        console.warn(`Error listing files in ${dirPath}: ${e.message}`);
        return [];
      }
    },
    
    /**
     * Safely load a file with error handling and fallback options
     * @param {string} path - Path to load
     * @param {string} moduleType - Type of module (vad, tts, kws, asr) for alternative paths
     * @param {object} options - Options for loading
     * @param {boolean} [options.tryAlternativePaths=true] - Whether to try alternative paths if first load fails
     * @param {any} [options.defaultValue=null] - Default value to return if loading fails
     * @returns {object} - Result object with success flag, data, and error message
     */
    safeLoadFile: function(path, moduleType, options = {}) {
      const { tryAlternativePaths = true, defaultValue = null } = options;
      let result = {
        success: false,
        data: defaultValue,
        error: null
      };
      
      try {
        // First try loading from the original path
        if (this.fileExists(path)) {
          const data = global.Module.FS.readFile(path);
          result.success = true;
          result.data = data;
          console.log(`Successfully loaded file from: ${path}`);
          return result;
        }
        
        // If the file doesn't exist at the original path and we should try alternatives
        if (tryAlternativePaths && moduleType) {
          // Extract filename from path
          const filename = path.split('/').pop();
          const alternativePath = this.getAssetPath(moduleType, filename);
          
          if (this.fileExists(alternativePath) && alternativePath !== path) {
            const data = global.Module.FS.readFile(alternativePath);
            result.success = true;
            result.data = data;
            console.log(`Loaded file from alternative path: ${alternativePath}`);
            return result;
          }
        }
        
        // If we get here, we couldn't find the file anywhere
        result.error = `File not found at path: ${path} or any alternative locations`;
        console.warn(result.error);
        return result;
      } catch (error) {
        result.error = `Error loading file: ${error.message || error}`;
        console.error(result.error);
        return result;
      }
    },
    
    /**
     * Safely load a file from a URL into the WASM file system
     * @param {string} url - URL to fetch the file from
     * @param {string} localPath - Path where to save the file in WASM filesystem
     * @param {boolean} debug - Whether to output debug logs
     * @returns {Promise<Object>} - Info about the loaded file
     */
    loadFile: async function(url, localPath, debug = false) {
      try {
        if (debug) console.log(`Loading file from ${url} to ${localPath}`);
        
        // Create parent directory if needed
        const lastSlash = localPath.lastIndexOf('/');
        if (lastSlash > 0) {
          const dirPath = localPath.substring(0, lastSlash);
          this.ensureDirectory(dirPath);
        }
        
        // Fetch the file
        if (debug) console.log(`Fetching ${url}`);
        const response = await fetch(url);
        
        if (!response.ok) {
          throw new Error(`Failed to fetch ${url}: ${response.status} ${response.statusText}`);
        }
        
        const buffer = await response.arrayBuffer();
        
        if (!buffer || buffer.byteLength === 0) {
          throw new Error(`Empty response from ${url}`);
        }
        
        if (debug) console.log(`Downloaded ${url}, size: ${buffer.byteLength} bytes`);
        
        // Write the file
        global.Module.FS.writeFile(localPath, new Uint8Array(buffer));
        
        return {
          success: true,
          path: localPath
        };
      } catch (error) {
        console.error(`Error loading ${url}:`, error);
        return {
          success: false,
          error: error.message
        };
      }
    },
    
    /**
     * Create directory and parents if needed
     * @param {string} dirPath - Directory path
     */
    ensureDirectory: function(dirPath) {
      if (!dirPath) return;
      
      // Skip if it's the root directory
      if (dirPath === '/') return;
      
      try {
        // Check if directory exists
        const stat = global.Module.FS.stat(dirPath);
        if (stat.isDir) return; // Already exists
        throw new Error(`Path exists but is not a directory: ${dirPath}`);
      } catch (error) {
        // If error is that the path doesn't exist, create it
        if (error.errno === 44 || error.errno === 2 || error.message.includes('No such file or directory')) {
          // Ensure parent directory exists first
          const parentDir = dirPath.substring(0, dirPath.lastIndexOf('/'));
          if (parentDir) this.ensureDirectory(parentDir);
          
          // Create this directory
          global.Module.FS.mkdir(dirPath);
          return;
        }
        
        // For other errors, rethrow
        throw error;
      }
    },
    
    /**
     * Extract a zip file to the WASM filesystem
     * @param {ArrayBuffer} zipData - The zip file data
     * @param {string} targetPath - Target extraction path
     * @param {boolean} debug - Enable debug logging
     * @returns {Promise<Object>} - Result of extraction
     */
    extractZip: async function(zipData, targetPath, debug = false) {
      if (debug) console.log(`Extracting zip to ${targetPath}`);
      
      try {
        // Make sure the base directory exists
        this.ensureDirectory(targetPath);
        
        // Load JSZip from CDN if needed
        if (typeof JSZip === 'undefined') {
          if (debug) console.log("Loading JSZip library from CDN");
          await new Promise((resolve, reject) => {
            const script = document.createElement('script');
            script.src = 'https://cdnjs.cloudflare.com/ajax/libs/jszip/3.10.1/jszip.min.js';
            script.onload = resolve;
            script.onerror = reject;
            document.head.appendChild(script);
          });
          
          if (typeof JSZip === 'undefined') {
            throw new Error("Failed to load JSZip library");
          }
        }
        
        // Process the zip file
        const zip = await JSZip.loadAsync(zipData);
        const extractedFiles = [];
        
        // First, create all directories
        for (const path in zip.files) {
          const file = zip.files[path];
          
          if (file.dir) {
            this.ensureDirectory(`${targetPath}/${path}`);
          } else {
            const dirPath = path.substring(0, path.lastIndexOf('/'));
            if (dirPath) {
              this.ensureDirectory(`${targetPath}/${dirPath}`);
            }
          }
        }
        
        // Now extract all files
        for (const path in zip.files) {
          const file = zip.files[path];
          if (file.dir) continue; // Skip directories, already created
          
          try {
            // Create the full path
            const fullPath = `${targetPath}/${path}`;
            
            // Extract and write the file
            const content = await file.async('arraybuffer');
            global.Module.FS.writeFile(fullPath, new Uint8Array(content));
            extractedFiles.push(fullPath);
          } catch (fileErr) {
            console.error(`Error extracting file ${path}: ${fileErr.message}`);
          }
        }
        
        if (debug) console.log(`Successfully extracted ${extractedFiles.length} files`);
        return { success: true, files: extractedFiles };
      } catch (error) {
        console.error(`Error extracting zip: ${error.message}`);
        return { success: false, error: error.message };
      }
    },
    
    /**
     * Debug the filesystem
     * @param {string} [path="/"] - Path to list
     */
    debugFilesystem: function(path = "/") {
      try {
        console.log(`--- Filesystem contents of ${path} ---`);
        if (!global.Module || !global.Module.FS) {
          console.log("Module.FS not available");
          return;
        }
        
        const entries = this.listFiles(path);
        console.log(entries);
        
        // Show preloaded asset directories
        Object.values(SherpaOnnx.Config.assetPaths).forEach(assetPath => {
          if (this.fileExists(assetPath)) {
            console.log(`--- ${assetPath} contents ---`);
            console.log(this.listFiles(assetPath));
          }
        });
      } catch (err) {
        console.error("Error debugging filesystem:", err);
      }
    }
  };
  
  // Resource tracking for cleanup
  SherpaOnnx.Resources = {
    // List of active resources by type
    active: {
      asr: [],
      vad: [],
      tts: [],
      kws: [],
      speakers: [],
      enhancement: []
    },
    
    /**
     * Track a resource for later cleanup
     * @param {string} type - Resource type
     * @param {Object} resource - Resource to track
     * @returns {Object} The resource (for chaining)
     */
    track: function(type, resource) {
      if (this.active[type]) {
        this.active[type].push(resource);
      }
      return resource;
    },
    
    /**
     * Clean up resources of a specific type
     * @param {string} [type] - Resource type (if omitted, clean all types)
     */
    cleanup: function(type) {
      if (type) {
        // Clean up specific type
        if (this.active[type]) {
          this.active[type].forEach(resource => {
            if (resource && typeof resource.free === 'function') {
              resource.free();
            }
          });
          this.active[type] = [];
        }
      } else {
        // Clean up all types
        Object.keys(this.active).forEach(t => this.cleanup(t));
      }
    }
  };
  
  // For convenience, add alias methods
  SherpaOnnx.trackResource = SherpaOnnx.Resources.track.bind(SherpaOnnx.Resources);
  SherpaOnnx.cleanup = SherpaOnnx.Resources.cleanup.bind(SherpaOnnx.Resources);
  
  // Expose SherpaOnnx to the global object
  global.SherpaOnnx = SherpaOnnx;
})(typeof window !== 'undefined' ? window : global); 