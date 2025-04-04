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
     * Load a TTS model from URLs
     * @param {Object} modelConfig - Configuration for the model
     * @returns {Promise<Object>} - Information about the loaded model
     */
    loadModel: async function(modelConfig) {
      const debug = modelConfig.debug || false;
      const modelDir = modelConfig.modelDir || 'tts-models';
      
      if (debug) console.log(`TTS.loadModel: Starting with base dir ${modelDir}`);
      
      try {
        // Always use clean start to avoid conflicts
        if (debug) console.log(`Cleaning model directory to prevent conflicts: ${modelDir}`);
        SherpaOnnx.FileSystem.removePath(modelDir, debug);
        
        // Flag to track if we need espeak data
        let needsEspeakData = false;
        
        // Prepare file list based on model type
        const files = [];
        
        if (modelConfig.type === 'vits') {
          // Add model file
          files.push({
            url: modelConfig.model || 'assets/tts/model.onnx',
            filename: 'model.onnx'
          });
          
          // Add tokens file
          files.push({
            url: modelConfig.tokens || 'assets/tts/tokens.txt',
            filename: 'tokens.txt'
          });
          
          // Add lexicon if provided
          if (modelConfig.lexicon) {
            files.push({
              url: modelConfig.lexicon,
              filename: 'lexicon.txt'
            });
          }
          
          // Flag that we need espeak-ng-data
          if (debug) console.log("Will load espeak-ng-data after model directory creation");
          needsEspeakData = true;
        } else if (modelConfig.type === 'matcha') {
          // Add required files for matcha
          files.push({
            url: modelConfig.acousticModel || 'assets/tts/acoustic_model.onnx',
            filename: 'acoustic_model.onnx'
          });
          
          files.push({
            url: modelConfig.vocoder || 'assets/tts/vocoder.onnx',
            filename: 'vocoder.onnx'
          });
          
          files.push({
            url: modelConfig.tokens || 'assets/tts/tokens.txt',
            filename: 'tokens.txt'
          });
          
          if (modelConfig.lexicon) {
            files.push({
              url: modelConfig.lexicon,
              filename: 'lexicon.txt'
            });
          }
        } else if (modelConfig.type === 'kokoro') {
          // Add required files for kokoro
          files.push({
            url: modelConfig.model || 'assets/tts/kokoro/model.onnx',
            filename: 'kokoro_model.onnx'
          });
          
          files.push({
            url: modelConfig.tokens || 'assets/tts/kokoro/tokens.txt',
            filename: 'tokens.txt'
          });
          
          if (modelConfig.voices) {
            files.push({
              url: modelConfig.voices,
              filename: 'voices.txt'
            });
          }
        }
        
        if (debug) console.log(`Prepared ${files.length} files to load for TTS model`);
        
        // Create unique model directory and load files
        const result = await SherpaOnnx.FileSystem.prepareModelDirectory(
          files, 
          modelDir, 
          debug
        );
        
        if (!result.success) {
          console.error("Failed to load model files:", result);
          throw new Error("Failed to load TTS model files");
        }
        
        // Handle espeak-ng-data for VITS models
        if (modelConfig.type === 'vits' && needsEspeakData) {
          if (debug) console.log(`Loading espeak-ng-data.zip into ${result.modelDir}`);
          
          try {
            // Use configurable URL if provided, otherwise use default
            const espeakZipUrl = modelConfig.espeakDataZip || 'assets/tts/espeak-ng-data.zip';
            if (debug) console.log(`Fetching espeak-ng-data from ${espeakZipUrl}`);
            
            const zipResponse = await fetch(espeakZipUrl);
            const zipData = await zipResponse.arrayBuffer();
            
            await SherpaOnnx.FileSystem.extractZip(
              zipData,
              result.modelDir,
              debug
            );
          } catch (zipError) {
            console.error("Error processing espeak-ng-data.zip:", zipError);
          }
        }
        
        // Organize files by type
        const modelFiles = {};
        const successFiles = result.files.filter(f => f.success);
        
        if (debug) console.log(`Successfully loaded ${successFiles.length} of ${result.files.length} files`);
        
        // Map files to their proper keys
        successFiles.forEach(file => {
          const filename = file.original.filename;
          
          if (filename === 'model.onnx') modelFiles.model = file.path;
          else if (filename === 'acoustic_model.onnx') modelFiles.acousticModel = file.path;
          else if (filename === 'vocoder.onnx') modelFiles.vocoder = file.path;
          else if (filename === 'tokens.txt') modelFiles.tokens = file.path;
          else if (filename === 'lexicon.txt') modelFiles.lexicon = file.path;
          else if (filename === 'voices.txt') modelFiles.voices = file.path;
          else if (filename === 'kokoro_model.onnx') modelFiles.kokoroModel = file.path;
        });
        
        // Return the model information
        return {
          modelDir: result.modelDir,
          type: modelConfig.type,
          files: modelFiles
        };
      } catch(e) {
        console.error(`TTS.loadModel: Error loading model:`, e);
        throw e;
      }
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
      
      let config = null;
      
      if (loadedModel.type === 'vits') {
        if (!loadedModel.files || !loadedModel.files.model || !loadedModel.files.tokens) {
          throw new Error("Missing required files for VITS model configuration");
        }
        
        const offlineTtsVitsModelConfig = {
          model: loadedModel.files.model,
          lexicon: loadedModel.files.lexicon || '',
          tokens: loadedModel.files.tokens,
          dataDir: `${loadedModel.modelDir}/espeak-ng-data`, // Path to espeak-ng-data in model directory
          dictDir: '',
          noiseScale: options.noiseScale || 0.667,
          noiseScaleW: options.noiseScaleW || 0.8,
          lengthScale: options.lengthScale || 1.0,
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
          numThreads: options.numThreads || 1,
          debug: debug ? 1 : 0,
          provider: 'cpu',
        };
        
        config = {
          offlineTtsModelConfig: offlineTtsModelConfig,
          ruleFsts: '',
          ruleFars: '',
          maxNumSentences: 1,
          silenceScale: options.silenceScale || 1.0
        };
      } else if (loadedModel.type === 'matcha') {
        // Similar configuration for matcha...
        // (Omitted for brevity)
      } else if (loadedModel.type === 'kokoro') {
        // Similar configuration for kokoro...
        // (Omitted for brevity)
      } else {
        throw new Error(`Unsupported TTS model type: ${loadedModel.type}`);
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
    if (!('offlineTtsVitsModelConfig' in config)) {
      config.offlineTtsVitsModelConfig = {
        model: './model.onnx',
        lexicon: '',
        tokens: './tokens.txt',
        dataDir: './espeak-ng-data',  // Use relative path in the model directory
        dictDir: '',
        noiseScale: 0.667,
        noiseScaleW: 0.8,
        lengthScale: 1.0,
      };
    }

    if (!('offlineTtsMatchaModelConfig' in config)) {
      config.offlineTtsMatchaModelConfig = {
        acousticModel: '',
        vocoder: '',
        lexicon: '',
        tokens: '',
        dataDir: '',
        dictDir: '',
        noiseScale: 0.667,
        lengthScale: 1.0,
      };
    }

    if (!('offlineTtsKokoroModelConfig' in config)) {
      config.offlineTtsKokoroModelConfig = {
        model: '',
        voices: '',
        tokens: '',
        lengthScale: 1.0,
        dataDir: '',
        dictDir: '',
        lexicon: '',
      };
    }

    const vitsModelConfig = initSherpaOnnxOfflineTtsVitsModelConfig(
      config.offlineTtsVitsModelConfig, Module);

    const matchaModelConfig = initSherpaOnnxOfflineTtsMatchaModelConfig(
      config.offlineTtsMatchaModelConfig, Module);

    const kokoroModelConfig = initSherpaOnnxOfflineTtsKokoroModelConfig(
      config.offlineTtsKokoroModelConfig, Module);

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
    const modelConfig = 
      initSherpaOnnxOfflineTtsModelConfig(config.offlineTtsModelConfig, Module);
    const len = modelConfig.len + 4 * 4;
    const ptr = Module._malloc(len);

    let offset = 0;
    Module._CopyHeap(modelConfig.ptr, modelConfig.len, ptr + offset);
    offset += modelConfig.len;

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
      buffer: buffer, ptr: ptr, len: len, config: modelConfig,
    };
  }
  
  /**
   * OfflineTts class for text-to-speech synthesis
   */
  global.OfflineTts = global.OfflineTts || function(configObj, Module) {
    if (Module.debug) {
      console.log("Creating OfflineTts with config:", JSON.stringify(configObj));
    }
    
    const config = initSherpaOnnxOfflineTtsConfig(configObj, Module);
    
    if (Module.debug) {
      try {
        Module._MyPrintTTS(config.ptr);
      } catch (e) {
        console.warn("Failed to print TTS config:", e);
      }
    }
    
    const handle = Module._SherpaOnnxCreateOfflineTts(config.ptr);
    
    if (!handle) {
      const error = new Error("Failed to create TTS engine - null handle returned");
      freeConfig(config, Module);
      throw error;
    }

    freeConfig(config, Module);

    this.handle = handle;
    this.sampleRate = Module._SherpaOnnxOfflineTtsSampleRate(this.handle);
    this.numSpeakers = Module._SherpaOnnxOfflineTtsNumSpeakers(this.handle);
    this.Module = Module;
    this.generatedAudios = []; // Track generated audios for cleanup
    
    /**
     * Generate speech from text
     * @param {string} text - Text to synthesize
     * @param {number} sid - Speaker ID (0 to numSpeakers-1)
     * @param {number} speed - Speed factor (1.0 is normal speed)
     * @returns {Object} - Object containing audio samples and sample rate
     */
    this.generate = function(text, sid = 0, speed = 1.0) {
      const textLen = this.Module.lengthBytesUTF8(text) + 1;
      const textPtr = this.Module._malloc(textLen);
      this.Module.stringToUTF8(text, textPtr, textLen);

      const h = this.Module._SherpaOnnxOfflineTtsGenerate(
        this.handle, textPtr, sid, speed);
      
      this.Module._free(textPtr);
      
      if (!h) {
        throw new Error("Failed to generate speech - null pointer returned");
      }

      const numSamples = this.Module.HEAP32[h / 4 + 1];
      const sampleRate = this.Module.HEAP32[h / 4 + 2];

      const samplesPtr = this.Module.HEAP32[h / 4] / 4;
      const samples = new Float32Array(numSamples);
      for (let i = 0; i < numSamples; i++) {
        samples[i] = this.Module.HEAPF32[samplesPtr + i];
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