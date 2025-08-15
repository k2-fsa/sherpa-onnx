/**
 * sherpa-onnx-tts.js
 * 
 * Text-to-Speech functionality for SherpaOnnx
 * Requires sherpa-onnx-core.js to be loaded first
 */

(function(global) {
  // Ensure the namespace exists
  if (!global.SherpaOnnx) {
    console.error('SherpaOnnx namespace not found. Make sure to load sherpa-onnx-core.js first.');
    return;
  }
  
  // Get a reference to the SherpaOnnx namespace
  const SherpaOnnx = global.SherpaOnnx;
  
  // Create or use existing TTS namespace
  SherpaOnnx.TTS = SherpaOnnx.TTS || {};
  
  // Define the TTS module functionality
  SherpaOnnx.TTS = {
    /**
     * Load a Text-to-Speech model
     * @param {Object} modelConfig - Configuration for the model
     * @returns {Promise<Object>} - Information about the loaded model
     */
    loadModel: async function(modelConfig) {
      const debug = modelConfig.debug || false;
      if (debug) console.log("TTS.loadModel: ModelConfig:", JSON.stringify(modelConfig));
      
      // Handle custom model upload case
      if (modelConfig.customModel) {
        if (debug) console.log("Using custom uploaded model");
        
        // Validate basic requirements
        if (!modelConfig.customModel.model && !modelConfig.customModel.acousticModel) {
          throw new Error("Missing required model file in custom model");
        }
        
        if (!modelConfig.customModel.tokens) {
          throw new Error("Missing required tokens.txt file in custom model");
        }
        
        return {
          modelDir: modelConfig.customModel.dataDir || 
                   (modelConfig.customModel.model 
                    ? modelConfig.customModel.model.split('/').slice(0, -1).join('/') 
                    : modelConfig.customModel.acousticModel.split('/').slice(0, -1).join('/')),
          modelType: modelConfig.modelType || 'vits',
          actualPaths: modelConfig.customModel,
          preloaded: false,
          options: modelConfig.options || {}
        };
      }
      
      // Default model directory and type handling
      const modelDir = modelConfig.modelDir || 'tts-models';
      const modelType = modelConfig.modelType || 'vits';
      
      // First check for preloaded assets
      if (!modelConfig.forceDownload) {
        const assetPath = SherpaOnnx.Config.assetPaths.tts;
        if (debug) console.log(`Checking for preloaded TTS assets at ${assetPath}`);
        
        if (SherpaOnnx.FileSystem.fileExists(assetPath)) {
          const files = SherpaOnnx.FileSystem.listFiles(assetPath);
          if (debug) console.log(`Found preloaded files: ${files.join(', ')}`);
          
          // Check for required model files based on type
          let hasRequiredFiles = false;
          const actualPaths = {};
          
          if (modelType === 'vits') {
            // VITS model requires model, lexicon, and tokens files
            const modelFile = files.find(f => f.endsWith('.onnx'));
            const tokensFile = files.find(f => f === 'tokens.txt');
            
            // Check for espeak data directory or zip
            let hasEspeakData = files.find(f => f === 'espeak-ng-data' || f === 'espeak-ng-data.zip');
            
            if (modelFile && tokensFile) {
              hasRequiredFiles = true;
              actualPaths.model = `${assetPath}/${modelFile}`;
              actualPaths.tokens = `${assetPath}/${tokensFile}`;
              
              // Add espeak data if found
              if (hasEspeakData) {
                if (hasEspeakData === 'espeak-ng-data') {
                  actualPaths.dataDir = `${assetPath}/espeak-ng-data`;
                } else {
                  // Will need to extract this later
                  actualPaths.espeakZip = `${assetPath}/espeak-ng-data.zip`;
                }
              }
            }
          }
          
          if (hasRequiredFiles) {
            if (debug) console.log("Using preloaded TTS model with paths:", actualPaths);
            return {
              modelDir: assetPath,
              modelType,
              actualPaths,
              preloaded: true,
              options: modelConfig.options || {}
            };
          }
          
          if (debug) console.log("Preloaded TTS assets found but missing required files");
        } else if (debug) {
          console.log(`Asset path ${assetPath} not found, will need to download models`);
        }
        
        // Also check alternative locations for preloaded assets
        const alternativePaths = [
          `/sherpa_assets/tts`,
          `/assets/tts`,
          `/preloaded/tts`
        ];
        
        for (const altPath of alternativePaths) {
          if (altPath === assetPath) continue; // Skip if we've already checked this path
          
          if (debug) console.log(`Checking alternative path: ${altPath}`);
          
          if (SherpaOnnx.FileSystem.fileExists(altPath)) {
            const files = SherpaOnnx.FileSystem.listFiles(altPath);
            if (debug) console.log(`Found files at ${altPath}: ${files.join(', ')}`);
            
            // Similar check for required files
            let hasRequiredFiles = false;
            const actualPaths = {};
            
            if (modelType === 'vits') {
              const modelFile = files.find(f => f.endsWith('.onnx'));
              const tokensFile = files.find(f => f === 'tokens.txt');
              
              // Check for espeak data directory or zip
              let hasEspeakData = files.find(f => f === 'espeak-ng-data' || f === 'espeak-ng-data.zip');
              
              if (modelFile && tokensFile) {
                hasRequiredFiles = true;
                actualPaths.model = `${altPath}/${modelFile}`;
                actualPaths.tokens = `${altPath}/${tokensFile}`;
                
                // Add espeak data if found
                if (hasEspeakData) {
                  if (hasEspeakData === 'espeak-ng-data') {
                    actualPaths.dataDir = `${altPath}/espeak-ng-data`;
                  } else {
                    // Will need to extract this later
                    actualPaths.espeakZip = `${altPath}/espeak-ng-data.zip`;
                  }
                }
              }
            }
            
            if (hasRequiredFiles) {
              if (debug) console.log(`Using alternative preloaded TTS model path: ${altPath}`);
              // Update the config to use this path in the future
              SherpaOnnx.Config.assetPaths.tts = altPath;
              
              return {
                modelDir: altPath,
                modelType,
                actualPaths,
                preloaded: true,
                options: modelConfig.options || {}
              };
            }
          }
        }
      }
      
      // If we reached here, we couldn't find preloaded assets
      throw new Error("No preloaded TTS model found and dynamic loading is not implemented");
    },
    
    /**
     * Create a TTS engine with the loaded model
     * @param {Object} loadedModel - Model information returned by loadModel
     * @param {Object} options - Additional configuration options
     * @returns {OfflineTts} - An instance of OfflineTts
     */
    createOfflineTts: function(loadedModel, options = {}) {
      const debug = options.debug !== undefined ? options.debug : false;
      
      if (debug) {
        console.log("Creating TTS engine with loaded model:", loadedModel);
      }
      
      // Always use a single consistent property name for model type
      const modelType = loadedModel.modelType || 'vits';
      
      if (debug) {
        console.log(`Using model type: ${modelType}`);
      }
      
      // Merge options from loadedModel with function options, prioritizing function options
      const mergedOptions = {
        ...loadedModel.options,
        ...options
      };
      
      if (debug) {
        console.log("Using merged options:", mergedOptions);
      }
      
      let config = null;
      
      if (modelType === 'vits') {
        // For preloaded assets, we use actualPaths
        const paths = loadedModel.actualPaths || loadedModel.files || {};
        
        if (debug) {
          console.log("Using model paths:", paths);
        }
        
        if (!paths.model || !paths.tokens) {
          throw new Error("Missing required files for VITS model configuration");
        }
        
        const offlineTtsVitsModelConfig = {
          model: paths.model,
          lexicon: paths.lexicon || '',
          tokens: paths.tokens,
          dataDir: paths.dataDir || `${loadedModel.modelDir}/espeak-ng-data`, // Path to espeak-ng-data in model directory
          dictDir: paths.dictDir || '',
          noiseScale: mergedOptions.noiseScale || 0.667,
          noiseScaleW: mergedOptions.noiseScaleW || 0.8,
          lengthScale: mergedOptions.lengthScale || 1.0,
        };
        
        if (debug) {
          console.log("VITS model config:", offlineTtsVitsModelConfig);
        }
        
        const offlineTtsMatchaModelConfig = {
          acousticModel: paths.acousticModel || '',
          vocoder: paths.vocoder || '',
          lexicon: paths.lexicon || '',
          tokens: paths.tokens || '',
          dataDir: paths.dataDir || '',
          dictDir: paths.dictDir || '',
          noiseScale: mergedOptions.noiseScale || 0.667,
          lengthScale: mergedOptions.lengthScale || 1.0,
        };
        
        const offlineTtsKokoroModelConfig = {
          model: paths.model || '',
          voices: paths.voices || '',
          tokens: paths.tokens || '',
          dataDir: paths.dataDir || '',
          lengthScale: mergedOptions.lengthScale || 1.0,
          dictDir: paths.dictDir || '',
          lexicon: paths.lexicon || '',
        };
        
        // Use the correct field names expected by the C API
        const offlineTtsModelConfig = {
          offlineTtsVitsModelConfig: offlineTtsVitsModelConfig,
          offlineTtsMatchaModelConfig: offlineTtsMatchaModelConfig,
          offlineTtsKokoroModelConfig: offlineTtsKokoroModelConfig,
          numThreads: mergedOptions.numThreads || 1,
          debug: debug ? 1 : 0,
          provider: mergedOptions.provider || 'cpu',
        };
        
        config = {
          offlineTtsModelConfig: offlineTtsModelConfig,
          ruleFsts: mergedOptions.ruleFsts || '',
          ruleFars: mergedOptions.ruleFars || '',
          maxNumSentences: mergedOptions.maxNumSentences || 1,
          silenceScale: mergedOptions.silenceScale || 1.0
        };
      } else if (modelType === 'matcha') {
        // Similar configuration for matcha...
        const paths = loadedModel.actualPaths || loadedModel.files || {};
        
        if (!paths.acousticModel || !paths.vocoder || !paths.tokens) {
          throw new Error("Missing required files for Matcha model configuration");
        }
        
        const offlineTtsVitsModelConfig = {
          model: '',
          lexicon: '',
          tokens: '',
          dataDir: '',
          dictDir: '',
          noiseScale: 0.667,
          noiseScaleW: 0.8,
          lengthScale: 1.0,
        };
        
        const offlineTtsMatchaModelConfig = {
          acousticModel: paths.acousticModel,
          vocoder: paths.vocoder,
          lexicon: paths.lexicon || '',
          tokens: paths.tokens,
          dataDir: paths.dataDir || '',
          dictDir: paths.dictDir || '',
          noiseScale: mergedOptions.noiseScale || 0.667,
          lengthScale: mergedOptions.lengthScale || 1.0,
        };
        
        const offlineTtsKokoroModelConfig = {
          model: '',
          voices: '',
          tokens: '',
          dataDir: '',
          lengthScale: 1.0,
          dictDir: '',
          lexicon: '',
        };
        
        const offlineTtsModelConfig = {
          offlineTtsVitsModelConfig: offlineTtsVitsModelConfig,
          offlineTtsMatchaModelConfig: offlineTtsMatchaModelConfig,
          offlineTtsKokoroModelConfig: offlineTtsKokoroModelConfig,
          numThreads: mergedOptions.numThreads || 1,
          debug: debug ? 1 : 0,
          provider: mergedOptions.provider || 'cpu',
        };
        
        config = {
          offlineTtsModelConfig: offlineTtsModelConfig,
          ruleFsts: mergedOptions.ruleFsts || '',
          ruleFars: mergedOptions.ruleFars || '',
          maxNumSentences: mergedOptions.maxNumSentences || 1,
          silenceScale: mergedOptions.silenceScale || 1.0
        };
      } else if (modelType === 'kokoro') {
        // Similar configuration for kokoro...
        const paths = loadedModel.actualPaths || loadedModel.files || {};
        
        if (!paths.model || !paths.voices || !paths.tokens) {
          throw new Error("Missing required files for Kokoro model configuration");
        }
        
        const offlineTtsVitsModelConfig = {
          model: '',
          lexicon: '',
          tokens: '',
          dataDir: '',
          dictDir: '',
          noiseScale: 0.667,
          noiseScaleW: 0.8,
          lengthScale: 1.0,
        };
        
        const offlineTtsMatchaModelConfig = {
          acousticModel: '',
          vocoder: '',
          lexicon: '',
          tokens: '',
          dataDir: '',
          dictDir: '',
          noiseScale: 0.667,
          lengthScale: 1.0,
        };
        
        const offlineTtsKokoroModelConfig = {
          model: paths.model,
          voices: paths.voices,
          tokens: paths.tokens,
          dataDir: paths.dataDir || '',
          lengthScale: mergedOptions.lengthScale || 1.0,
          dictDir: paths.dictDir || '',
          lexicon: paths.lexicon || '',
        };
        
        const offlineTtsModelConfig = {
          offlineTtsVitsModelConfig: offlineTtsVitsModelConfig,
          offlineTtsMatchaModelConfig: offlineTtsMatchaModelConfig,
          offlineTtsKokoroModelConfig: offlineTtsKokoroModelConfig,
          numThreads: mergedOptions.numThreads || 1,
          debug: debug ? 1 : 0,
          provider: mergedOptions.provider || 'cpu',
        };
        
        config = {
          offlineTtsModelConfig: offlineTtsModelConfig,
          ruleFsts: mergedOptions.ruleFsts || '',
          ruleFars: mergedOptions.ruleFars || '',
          maxNumSentences: mergedOptions.maxNumSentences || 1,
          silenceScale: mergedOptions.silenceScale || 1.0
        };
      } else {
        throw new Error(`Unsupported TTS model type: ${modelType}`);
      }
      
      if (debug) {
        console.log("Final TTS configuration:", JSON.stringify(config));
      }
      
      try {
        // Create the offline TTS object
        const tts = this.createOfflineTtsInternal(config, global.Module);
        
        // Track the resource for cleanup if tracking function is available
        if (SherpaOnnx.trackResource) {
          SherpaOnnx.trackResource('tts', tts);
        }
        
        return tts;
      } catch (error) {
        console.error("Error creating TTS engine:", error);
        throw error;
      }
    },
    
    /**
     * Internal function to create an offline TTS engine
     * Following the reference implementation pattern
     */
    createOfflineTtsInternal: function(config, Module) {
      if (!config) {
        console.error("TTS configuration is null or undefined");
        return null;
      }
      
      if (typeof createOfflineTts === 'function') {
        // Use the global createOfflineTts function if available
        return createOfflineTts(Module, config);
      }
      
      // Otherwise use our own implementation
      return new global.OfflineTts(config, Module);
    }
  };
  
  /**
   * Free configuration memory allocated in WASM
   * @param {Object} config - Configuration object with allocated memory
   * @param {Object} Module - WebAssembly module
   * @private
   */
  function freeConfig(config, Module) {
    if ('buffer' in config) {
      Module._free(config.buffer);
    }

    if ('config' in config) {
      freeConfig(config.config, Module);
    }

    if ('matcha' in config) {
      freeConfig(config.matcha, Module);
    }

    if ('kokoro' in config) {
      freeConfig(config.kokoro, Module);
    }

    if (config.ptr) {
      Module._free(config.ptr);
    }
  }
  
  /**
   * Initialize VITS model configuration
   * @param {Object} config - VITS configuration
   * @param {Object} Module - WebAssembly module
   * @returns {Object} - Configuration with pointers
   */
  function initSherpaOnnxOfflineTtsVitsModelConfig(config, Module) {
    const modelLen = Module.lengthBytesUTF8(config.model || '') + 1;
    const lexiconLen = Module.lengthBytesUTF8(config.lexicon || '') + 1;
    const tokensLen = Module.lengthBytesUTF8(config.tokens || '') + 1;
    const dataDirLen = Module.lengthBytesUTF8(config.dataDir || '') + 1;
    const dictDirLen = Module.lengthBytesUTF8(config.dictDir || '') + 1;

    const n = modelLen + lexiconLen + tokensLen + dataDirLen + dictDirLen;
    const buffer = Module._malloc(n);

    const len = 8 * 4;
    const ptr = Module._malloc(len);

    let offset = 0;
    Module.stringToUTF8(config.model || '', buffer + offset, modelLen);
    offset += modelLen;

    Module.stringToUTF8(config.lexicon || '', buffer + offset, lexiconLen);
    offset += lexiconLen;

    Module.stringToUTF8(config.tokens || '', buffer + offset, tokensLen);
    offset += tokensLen;

    Module.stringToUTF8(config.dataDir || '', buffer + offset, dataDirLen);
    offset += dataDirLen;

    Module.stringToUTF8(config.dictDir || '', buffer + offset, dictDirLen);
    offset += dictDirLen;

    offset = 0;
    Module.setValue(ptr, buffer + offset, 'i8*');
    offset += modelLen;

    Module.setValue(ptr + 4, buffer + offset, 'i8*');
    offset += lexiconLen;

    Module.setValue(ptr + 8, buffer + offset, 'i8*');
    offset += tokensLen;

    Module.setValue(ptr + 12, buffer + offset, 'i8*');
    offset += dataDirLen;

    Module.setValue(ptr + 16, config.noiseScale || 0.667, 'float');
    Module.setValue(ptr + 20, config.noiseScaleW || 0.8, 'float');
    Module.setValue(ptr + 24, config.lengthScale || 1.0, 'float');
    Module.setValue(ptr + 28, buffer + offset, 'i8*');
    offset += dictDirLen;

    return {
      buffer: buffer, ptr: ptr, len: len,
    };
  }
  
  /**
   * Initialize Matcha model configuration
   * @param {Object} config - Matcha configuration
   * @param {Object} Module - WebAssembly module
   * @returns {Object} - Configuration with pointers
   */
  function initSherpaOnnxOfflineTtsMatchaModelConfig(config, Module) {
    const acousticModelLen = Module.lengthBytesUTF8(config.acousticModel || '') + 1;
    const vocoderLen = Module.lengthBytesUTF8(config.vocoder || '') + 1;
    const lexiconLen = Module.lengthBytesUTF8(config.lexicon || '') + 1;
    const tokensLen = Module.lengthBytesUTF8(config.tokens || '') + 1;
    const dataDirLen = Module.lengthBytesUTF8(config.dataDir || '') + 1;
    const dictDirLen = Module.lengthBytesUTF8(config.dictDir || '') + 1;

    const n = acousticModelLen + vocoderLen + lexiconLen + tokensLen +
      dataDirLen + dictDirLen;

    const buffer = Module._malloc(n);
    const len = 8 * 4;
    const ptr = Module._malloc(len);

    let offset = 0;
    Module.stringToUTF8(
      config.acousticModel || '', buffer + offset, acousticModelLen);
    offset += acousticModelLen;

    Module.stringToUTF8(config.vocoder || '', buffer + offset, vocoderLen);
    offset += vocoderLen;

    Module.stringToUTF8(config.lexicon || '', buffer + offset, lexiconLen);
    offset += lexiconLen;

    Module.stringToUTF8(config.tokens || '', buffer + offset, tokensLen);
    offset += tokensLen;

    Module.stringToUTF8(config.dataDir || '', buffer + offset, dataDirLen);
    offset += dataDirLen;

    Module.stringToUTF8(config.dictDir || '', buffer + offset, dictDirLen);
    offset += dictDirLen;

    offset = 0;
    Module.setValue(ptr, buffer + offset, 'i8*');
    offset += acousticModelLen;

    Module.setValue(ptr + 4, buffer + offset, 'i8*');
    offset += vocoderLen;

    Module.setValue(ptr + 8, buffer + offset, 'i8*');
    offset += lexiconLen;

    Module.setValue(ptr + 12, buffer + offset, 'i8*');
    offset += tokensLen;

    Module.setValue(ptr + 16, buffer + offset, 'i8*');
    offset += dataDirLen;

    Module.setValue(ptr + 20, config.noiseScale || 0.667, 'float');
    Module.setValue(ptr + 24, config.lengthScale || 1.0, 'float');
    Module.setValue(ptr + 28, buffer + offset, 'i8*');
    offset += dictDirLen;

    return {
      buffer: buffer, ptr: ptr, len: len,
    };
  }
  
  /**
   * Initialize Kokoro model configuration
   * @param {Object} config - Kokoro configuration
   * @param {Object} Module - WebAssembly module
   * @returns {Object} - Configuration with pointers
   */
  function initSherpaOnnxOfflineTtsKokoroModelConfig(config, Module) {
    const modelLen = Module.lengthBytesUTF8(config.model || '') + 1;
    const voicesLen = Module.lengthBytesUTF8(config.voices || '') + 1;
    const tokensLen = Module.lengthBytesUTF8(config.tokens || '') + 1;
    const dataDirLen = Module.lengthBytesUTF8(config.dataDir || '') + 1;
    const dictDirLen = Module.lengthBytesUTF8(config.dictDir || '') + 1;
    const lexiconLen = Module.lengthBytesUTF8(config.lexicon || '') + 1;

    const n = modelLen + voicesLen + tokensLen + dataDirLen + dictDirLen + lexiconLen;
    const buffer = Module._malloc(n);

    const len = 7 * 4;
    const ptr = Module._malloc(len);

    let offset = 0;
    Module.stringToUTF8(config.model || '', buffer + offset, modelLen);
    offset += modelLen;

    Module.stringToUTF8(config.voices || '', buffer + offset, voicesLen);
    offset += voicesLen;

    Module.stringToUTF8(config.tokens || '', buffer + offset, tokensLen);
    offset += tokensLen;

    Module.stringToUTF8(config.dataDir || '', buffer + offset, dataDirLen);
    offset += dataDirLen;

    Module.stringToUTF8(config.dictDir || '', buffer + offset, dictDirLen);
    offset += dictDirLen;

    Module.stringToUTF8(config.lexicon || '', buffer + offset, lexiconLen);
    offset += lexiconLen;

    offset = 0;
    Module.setValue(ptr, buffer + offset, 'i8*');
    offset += modelLen;

    Module.setValue(ptr + 4, buffer + offset, 'i8*');
    offset += voicesLen;

    Module.setValue(ptr + 8, buffer + offset, 'i8*');
    offset += tokensLen;

    Module.setValue(ptr + 12, buffer + offset, 'i8*');
    offset += dataDirLen;

    Module.setValue(ptr + 16, config.lengthScale || 1.0, 'float');

    Module.setValue(ptr + 20, buffer + offset, 'i8*');
    offset += dictDirLen;

    Module.setValue(ptr + 24, buffer + offset, 'i8*');
    offset += lexiconLen;

    return {
      buffer: buffer, ptr: ptr, len: len,
    };
  }
  
  /**
   * Initialize offline TTS model configuration
   * @param {Object} config - Model configuration
   * @param {Object} Module - WebAssembly module
   * @returns {Object} - Configuration with pointers
   */
  function initSherpaOnnxOfflineTtsModelConfig(config, Module) {
    if (Module.debug) {
      console.log("Initializing offline TTS model config:", JSON.stringify(config));
    }
    
    // Get configurations, supporting both old and new formats
    const vitsConfig = config.vits || config.offlineTtsVitsModelConfig || {
      model: './model.onnx',
      lexicon: '',
      tokens: './tokens.txt',
      dataDir: './espeak-ng-data',  // Use relative path in the model directory
      dictDir: '',
      noiseScale: 0.667,
      noiseScaleW: 0.8,
      lengthScale: 1.0,
    };

    const matchaConfig = config.matcha || config.offlineTtsMatchaModelConfig || {
      acousticModel: '',
      vocoder: '',
      lexicon: '',
      tokens: '',
      dataDir: '',
      dictDir: '',
      noiseScale: 0.667,
      lengthScale: 1.0,
    };

    const kokoroConfig = config.kokoro || config.offlineTtsKokoroModelConfig || {
      model: '',
      voices: '',
      tokens: '',
      dataDir: '',
      lengthScale: 1.0,
      dictDir: '',
      lexicon: '',
    };

    const vitsModelConfig = initSherpaOnnxOfflineTtsVitsModelConfig(
      vitsConfig, Module);

    const matchaModelConfig = initSherpaOnnxOfflineTtsMatchaModelConfig(
      matchaConfig, Module);

    const kokoroModelConfig = initSherpaOnnxOfflineTtsKokoroModelConfig(
      kokoroConfig, Module);

    const len = vitsModelConfig.len + matchaModelConfig.len +
      kokoroModelConfig.len + 3 * 4;

    const ptr = Module._malloc(len);

    let offset = 0;
    Module._CopyHeap(vitsModelConfig.ptr, vitsModelConfig.len, ptr + offset);
    offset += vitsModelConfig.len;

    Module.setValue(ptr + offset, config.numThreads || 1, 'i32');
    offset += 4;

    Module.setValue(ptr + offset, config.debug || 0, 'i32');
    offset += 4;

    const providerLen = Module.lengthBytesUTF8(config.provider || 'cpu') + 1;
    const buffer = Module._malloc(providerLen);
    Module.stringToUTF8(config.provider || 'cpu', buffer, providerLen);
    Module.setValue(ptr + offset, buffer, 'i8*');
    offset += 4;

    Module._CopyHeap(matchaModelConfig.ptr, matchaModelConfig.len, ptr + offset);
    offset += matchaModelConfig.len;

    Module._CopyHeap(kokoroModelConfig.ptr, kokoroModelConfig.len, ptr + offset);
    offset += kokoroModelConfig.len;

    return {
      buffer: buffer, ptr: ptr, len: len, config: vitsModelConfig,
      matcha: matchaModelConfig, kokoro: kokoroModelConfig,
    };
  }
  
  /**
   * Initialize the TTS configuration
   * @param {Object} config - TTS configuration
   * @param {Object} Module - WebAssembly module
   * @returns {Object} - Configuration with pointers
   */
  function initSherpaOnnxOfflineTtsConfig(config, Module) {
    // Log for debugging
    if (Module.debug) {
      console.log("Initializing TTS config:", JSON.stringify(config));
    }
    
    // Make sure we have an offlineTtsModelConfig
    if (!config.offlineTtsModelConfig) {
      if (Module.debug) {
        console.log("No offlineTtsModelConfig found, creating default");
      }
      
      // Use provided defaults or create new ones
      config.offlineTtsModelConfig = {
        offlineTtsVitsModelConfig: {
          model: './model.onnx',
          lexicon: '',
          tokens: './tokens.txt',
          dataDir: './espeak-ng-data',
          dictDir: '',
          noiseScale: 0.667,
          noiseScaleW: 0.8,
          lengthScale: 1.0,
        },
        offlineTtsMatchaModelConfig: {
          acousticModel: '',
          vocoder: '',
          lexicon: '',
          tokens: '',
          dataDir: '',
          dictDir: '',
          noiseScale: 0.667,
          lengthScale: 1.0,
        },
        offlineTtsKokoroModelConfig: {
          model: '',
          voices: '',
          tokens: '',
          dataDir: '',
          lengthScale: 1.0,
          dictDir: '',
          lexicon: '',
        },
        numThreads: 1,
        debug: Module.debug ? 1 : 0,
        provider: 'cpu',
      };
    }
    
    // Initialize model config
    const initializedModelConfig = 
      initSherpaOnnxOfflineTtsModelConfig(config.offlineTtsModelConfig, Module);
    
    const len = initializedModelConfig.len + 4 * 4;
    const ptr = Module._malloc(len);

    let offset = 0;
    Module._CopyHeap(initializedModelConfig.ptr, initializedModelConfig.len, ptr + offset);
    offset += initializedModelConfig.len;

    const ruleFstsLen = Module.lengthBytesUTF8(config.ruleFsts || '') + 1;
    const ruleFarsLen = Module.lengthBytesUTF8(config.ruleFars || '') + 1;

    const buffer = Module._malloc(ruleFstsLen + ruleFarsLen);
    Module.stringToUTF8(config.ruleFsts || '', buffer, ruleFstsLen);
    Module.stringToUTF8(config.ruleFars || '', buffer + ruleFstsLen, ruleFarsLen);

    Module.setValue(ptr + offset, buffer, 'i8*');
    offset += 4;

    Module.setValue(ptr + offset, config.maxNumSentences || 1, 'i32');
    offset += 4;

    Module.setValue(ptr + offset, buffer + ruleFstsLen, 'i8*');
    offset += 4;

    Module.setValue(ptr + offset, config.silenceScale || 1.0, 'float');
    offset += 4;

    return {
      buffer: buffer, ptr: ptr, len: len, config: initializedModelConfig,
    };
  }
  
  /**
   * OfflineTts class for text-to-speech synthesis
   */
  global.OfflineTts = global.OfflineTts || function(configObj, Module) {
    if (!Module) {
      throw new Error("WASM Module is required for OfflineTts");
    }
    
    this.Module = Module;
    this.handle = null;
    this.sampleRate = 0;
    this.numSpeakers = 0;
    this.generatedAudios = []; // Track generated audios for cleanup
    
    const debug = Module.debug || (configObj && configObj.debug);
    
    if (debug) {
      console.log("Creating OfflineTts with config:", JSON.stringify(configObj));
    }
    
    try {
      // Initialize the TTS configuration
      const config = initSherpaOnnxOfflineTtsConfig(configObj, Module);
      
      if (debug) {
        try {
          Module._MyPrintTTS(config.ptr);
        } catch (e) {
          console.warn("Failed to print TTS config:", e);
        }
      }
      
      // Create the TTS engine
      const handle = Module._SherpaOnnxCreateOfflineTts(config.ptr);
      
      if (!handle || handle === 0) {
        const error = new Error("Failed to create TTS engine - null handle returned");
        freeConfig(config, Module);
        throw error;
      }
      
      // Free the configuration memory now that we have the handle
      freeConfig(config, Module);
      
      // Store the handle and get basic information about the TTS engine
      this.handle = handle;
      
      try {
        this.sampleRate = Module._SherpaOnnxOfflineTtsSampleRate(this.handle);
        this.numSpeakers = Module._SherpaOnnxOfflineTtsNumSpeakers(this.handle);
        
        if (debug) {
          console.log(`TTS engine initialized. Sample rate: ${this.sampleRate}Hz, Number of speakers: ${this.numSpeakers}`);
        }
      } catch (e) {
        console.error("Error getting TTS engine information:", e);
        // Don't throw here, we can continue with defaults
      }
    } catch (e) {
      // Clean up any resources if initialization failed
      if (this.handle) {
        try {
          Module._SherpaOnnxDestroyOfflineTts(this.handle);
        } catch (cleanupError) {
          console.error("Error cleaning up after failed initialization:", cleanupError);
        }
        this.handle = null;
      }
      
      // Re-throw the original error
      throw e;
    }
    
    /**
     * Generate speech from text
     * @param {string} text - Text to synthesize
     * @param {number} sid - Speaker ID (0 to numSpeakers-1)
     * @param {number} speed - Speed factor (1.0 is normal speed)
     * @returns {Object} - Object containing audio samples and sample rate
     */
    this.generate = function(text, sid = 0, speed = 1.0) {
      if (this.Module.debug) {
        console.log(`Generating speech for text: "${text}", sid: ${sid}, speed: ${speed}`);
      }
      
      const textLen = this.Module.lengthBytesUTF8(text) + 1;
      const textPtr = this.Module._malloc(textLen);
      this.Module.stringToUTF8(text, textPtr, textLen);

      const h = this.Module._SherpaOnnxOfflineTtsGenerate(
        this.handle, textPtr, sid, speed);
      
      this.Module._free(textPtr);
      
      if (!h || h === 0) {
        throw new Error("Failed to generate speech - null pointer returned");
      }

      // Access the generated audio structure
      // The structure has this format in C:
      // struct SherpaOnnxOfflineTtsGeneratedAudio {
      //   float *samples;
      //   int32_t n;
      //   int32_t sample_rate;
      // };
      try {
        // Read the number of samples and sample rate from memory
        const numSamples = this.Module.getValue(h + 4, 'i32');
        const sampleRate = this.Module.getValue(h + 8, 'i32');
        
        if (this.Module.debug) {
          console.log(`Generated ${numSamples} samples at ${sampleRate}Hz`);
        }
        
        // Get the pointer to the audio samples array
        const samplesPtr = this.Module.getValue(h, '*');
        
        if (!samplesPtr) {
          throw new Error("Failed to read audio samples pointer");
        }
        
        // Copy samples to a new Float32Array
        const samples = new Float32Array(numSamples);
        for (let i = 0; i < numSamples; i++) {
          samples[i] = this.Module.getValue(samplesPtr + (i * 4), 'float');
        }

        // Add to our tracking list
        this.generatedAudios.push(h);
        
        return {
          samples: samples, 
          sampleRate: sampleRate,
          // Add a cleanup function for this specific audio
          free: () => {
            const index = this.generatedAudios.indexOf(h);
            if (index !== -1) {
              this.Module._SherpaOnnxDestroyOfflineTtsGeneratedAudio(h);
              this.generatedAudios.splice(index, 1);
            }
          }
        };
      } catch (error) {
        // Clean up on error to avoid memory leaks
        if (h) {
          this.Module._SherpaOnnxDestroyOfflineTtsGeneratedAudio(h);
        }
        console.error("Error accessing generated audio data:", error);
        throw new Error("Failed to process generated audio: " + error.message);
      }
    };
    
    /**
     * Save generated audio to a WAV file (for browser environments)
     * @param {Float32Array} samples - Audio samples
     * @param {number} sampleRate - Sample rate
     * @returns {Blob} - WAV file as Blob
     */
    this.saveAsWav = function(samples, sampleRate) {
      // Create WAV file in memory
      const numSamples = samples.length;
      const dataSize = numSamples * 2; // 16-bit samples
      const bufferSize = 44 + dataSize;
      
      const buffer = new ArrayBuffer(bufferSize);
      const view = new DataView(buffer);
      
      // WAV header (http://soundfile.sapp.org/doc/WaveFormat/)
      view.setUint32(0, 0x46464952, true); // 'RIFF'
      view.setUint32(4, bufferSize - 8, true); // chunk size
      view.setUint32(8, 0x45564157, true); // 'WAVE'
      view.setUint32(12, 0x20746d66, true); // 'fmt '
      view.setUint32(16, 16, true); // subchunk1 size
      view.setUint16(20, 1, true); // PCM format
      view.setUint16(22, 1, true); // mono
      view.setUint32(24, sampleRate, true); // sample rate
      view.setUint32(28, sampleRate * 2, true); // byte rate
      view.setUint16(32, 2, true); // block align
      view.setUint16(34, 16, true); // bits per sample
      view.setUint32(36, 0x61746164, true); // 'data'
      view.setUint32(40, dataSize, true); // subchunk2 size
      
      // Write audio data
      for (let i = 0; i < numSamples; i++) {
        // Convert float to 16-bit PCM
        let sample = samples[i];
        if (sample > 1.0) sample = 1.0;
        if (sample < -1.0) sample = -1.0;
        
        const pcm = Math.floor(sample * 32767);
        view.setInt16(44 + i * 2, pcm, true);
      }
      
      return new Blob([buffer], { type: 'audio/wav' });
    };
    
    /**
     * Free the TTS engine and all generated audios
     */
    this.free = function() {
      // Free all generated audios first
      for (let i = this.generatedAudios.length - 1; i >= 0; i--) {
        if (this.generatedAudios[i]) {
          this.Module._SherpaOnnxDestroyOfflineTtsGeneratedAudio(this.generatedAudios[i]);
        }
      }
      this.generatedAudios = [];
      
      // Free the TTS engine
      if (this.handle) {
        this.Module._SherpaOnnxDestroyOfflineTts(this.handle);
        this.handle = 0;
      }
    };
  };
  
  // For Node.js environments
  if (typeof module !== 'undefined' && module.exports) {
    module.exports = SherpaOnnx;
  }
})(typeof window !== 'undefined' ? window : global);

/**
 * Global helper function to create an OfflineTts instance
 */
function createOfflineTts(Module, config) {
  // Use provided config or create default
  if (config) return new OfflineTts(config, Module);
  
  // Default configuration pointing to extracted espeak-ng-data
  const defaultConfig = {
    offlineTtsModelConfig: {
      offlineTtsVitsModelConfig: {
        model: './model.onnx',
        lexicon: '',
        tokens: './tokens.txt',
        dataDir: './espeak-ng-data',  // Use relative path in the model directory
        dictDir: '',
        noiseScale: 0.667,
        noiseScaleW: 0.8,
        lengthScale: 1.0,
      },
      offlineTtsMatchaModelConfig: {
        acousticModel: '',
        vocoder: '',
        lexicon: '',
        tokens: '',
        dataDir: '',
        dictDir: '',
        noiseScale: 0.667,
        lengthScale: 1.0,
      },
      offlineTtsKokoroModelConfig: {
        model: '',
        voices: '',
        tokens: '',
        dataDir: '',
        lengthScale: 1.0,
        dictDir: '',
        lexicon: '',
      },
      numThreads: 1,
      debug: 1,
      provider: 'cpu',
    },
    ruleFsts: '',
    ruleFars: '',
    maxNumSentences: 1,
    silenceScale: 1.0
  };
  
  return new OfflineTts(defaultConfig, Module);
} 