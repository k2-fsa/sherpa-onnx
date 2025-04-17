/**
 * sherpa-onnx-vad.js
 * 
 * Voice Activity Detection functionality for SherpaOnnx
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
  
  // Internal class for voice activity detection
  class VoiceActivityDetector {
    constructor(handle, Module) {
      this.handle = handle;
      this.Module = Module;
    }
    
    /**
     * Accept audio waveform data
     * @param {Float32Array} samples - Audio samples in [-1, 1] range
     */
    acceptWaveform(samples) {
      const pointer = this.Module._malloc(samples.length * samples.BYTES_PER_ELEMENT);
      this.Module.HEAPF32.set(samples, pointer / samples.BYTES_PER_ELEMENT);
      
      this.Module._SherpaOnnxVoiceActivityDetectorAcceptWaveform(
        this.handle, pointer, samples.length
      );
      
      this.Module._free(pointer);
    }
    
    /**
     * Check if there are no speech segments available
     * @returns {boolean} - True if no segments available, false otherwise
     */
    isEmpty() {
      return this.Module._SherpaOnnxVoiceActivityDetectorEmpty(this.handle) === 1;
    }
    
    /**
     * Check if voice is detected
     * @returns {boolean} - True if voice detected, false otherwise
     */
    detected() {
      return this.Module._SherpaOnnxVoiceActivityDetectorDetected(this.handle) === 1;
    }
    
    /**
     * Reset the detector
     */
    reset() {
      this.Module._SherpaOnnxVoiceActivityDetectorReset(this.handle);
    }
    
    /**
     * Free the detector
     */
    free() {
      if (this.handle) {
        this.Module._SherpaOnnxDestroyVoiceActivityDetector(this.handle);
        this.handle = 0;
      }
    }
  }
  
  // Define the VAD module functionality
  SherpaOnnx.VAD = {
    /**
     * Load a VAD model from URL
     * @param {Object} modelConfig - Configuration for the model
     * @returns {Promise<Object>} - Information about the loaded model
     */
    loadModel: async function(modelConfig) {
      const debug = modelConfig.debug !== false;
      
      if (debug) console.log("VAD.loadModel: ModelConfig received:", JSON.stringify(modelConfig));
      
      // Use configurable model directory with default
      const modelDir = modelConfig.modelDir || 'vad-models';
      const fileName = modelConfig.fileName || 'silero_vad.onnx';
      const destPath = `${modelDir}/${fileName}`;
      
      if (debug) console.log(`VAD.loadModel: Using model directory: ${modelDir}`);
      if (debug) console.log(`VAD.loadModel: Target file path: ${destPath}`);
      
      try {
        // Clean up existing path if needed for fresh start
        if (modelConfig.cleanStart) {
          if (debug) console.log(`VAD.loadModel: Clean start requested, removing existing paths`);
          SherpaOnnx.FileSystem.removePath(modelDir, debug);
        }
        
        // Load the model using the safe loader
        if (debug) console.log(`VAD.loadModel: Loading model from ${modelConfig.model || 'assets/vad/silero_vad.onnx'} to ${destPath}`);
        
        const loadResult = await SherpaOnnx.FileSystem.safeLoadFile(
          modelConfig.model || 'assets/vad/silero_vad.onnx', 
          destPath,
          debug
        );
        
        if (!loadResult || (typeof loadResult === 'object' && !loadResult.success)) {
          throw new Error(`Failed to load model from ${modelConfig.model || 'assets/vad/silero_vad.onnx'} to ${destPath}`);
        }
        
        // Update the path if it was changed
        const actualPath = (typeof loadResult === 'object' && loadResult.path) ? loadResult.path : destPath;
        if (actualPath !== destPath) {
          if (debug) console.log(`VAD.loadModel: Note - Model loaded to alternate path: ${actualPath}`);
        }
        
        if (debug) console.log(`VAD.loadModel: Model loaded successfully`);
        
        // Return model information with the actual path used
        return { 
          modelDir,
          fileName,
          modelPath: actualPath
        };
      } catch (error) {
        console.error(`VAD.loadModel: Error loading model:`, error);
        throw error;
      }
    },
    
    /**
     * Create a Voice Activity Detector with the loaded model
     * @param {Object} loadedModel - Model information returned by loadModel
     * @param {Object} options - Additional configuration options
     * @returns {VoiceActivityDetector} - A VAD instance
     */
    createVoiceActivityDetector: function(loadedModel, options = {}) {
      const debug = options.debug !== false;
      
      try {
        // Get the model path from loaded model info
        const modelPath = loadedModel.modelPath || `${loadedModel.modelDir}/${loadedModel.fileName || 'silero_vad.onnx'}`;
        
        if (debug) console.log(`VAD.createVoiceActivityDetector: Using model at ${modelPath}`);
        
        // Verify model file exists before proceeding
        if (!SherpaOnnx.FileSystem.fileExists(modelPath)) {
          throw new Error(`Model file not found at ${modelPath}`);
        }
        
        // Initialize the silero VAD config
        const sileroVadConfig = this._initSileroVadConfig({
          model: modelPath,
          threshold: options.threshold || 0.5,
          minSilenceDuration: options.minSilenceDuration || 0.3,
          minSpeechDuration: options.minSpeechDuration || 0.1,
          windowSize: options.windowSize || 512,
          maxSpeechDuration: options.maxSpeechDuration || 30.0,
        }, global.Module);
        
        // Initialize the full VAD config
        const vadConfig = this._initVadModelConfig({
          sileroVad: {
            model: modelPath,
            threshold: options.threshold || 0.5,
            minSilenceDuration: options.minSilenceDuration || 0.3,
            minSpeechDuration: options.minSpeechDuration || 0.1,
            windowSize: options.windowSize || 512,
            maxSpeechDuration: options.maxSpeechDuration || 30.0,
          },
          sampleRate: options.sampleRate || 16000,
          numThreads: options.numThreads || 1,
          provider: options.provider || 'cpu',
          debug: debug ? 1 : 0,
        }, global.Module);
        
        // Debug print the config if requested
        if (debug) {
          try {
            global.Module._MyPrintVAD(vadConfig.ptr);
          } catch (printErr) {
            console.warn("Could not print VAD config:", printErr);
          }
        }
        
        // Create the detector
        if (debug) console.log("VAD.createVoiceActivityDetector: Creating detector");
        const vadPtr = global.Module.ccall(
          'SherpaOnnxCreateVoiceActivityDetector',
          'number',
          ['number', 'number'],
          [vadConfig.ptr, options.bufferSizeInSeconds || 5.0]
        );
        
        if (!vadPtr) {
          throw new Error("Failed to create voice activity detector");
        }
        
        if (debug) console.log("VAD.createVoiceActivityDetector: Detector created successfully");
        
        // Free configuration memory
        SherpaOnnx.Utils.freeConfig(vadConfig, global.Module);
        
        return new VoiceActivityDetector(vadPtr, global.Module);
      } catch (error) {
        console.error("Error creating VAD detector:", error);
        throw error;
      }
    },
    
    /**
     * Initialize SileroVad configuration in WASM
     * @param {Object} config - SileroVad configuration
     * @param {Object} Module - WebAssembly module
     * @returns {Object} Configuration with WASM pointers
     * @private
     */
    _initSileroVadConfig: function(config, Module) {
      const modelString = SherpaOnnx.Utils.allocateString(config.model, Module);
      
      const len = 6 * 4;
      const ptr = Module._malloc(len);
      
      let offset = 0;
      Module.setValue(ptr, modelString.ptr, 'i8*');
      offset += 4;
      
      Module.setValue(ptr + offset, config.threshold || 0.5, 'float');
      offset += 4;
      
      Module.setValue(ptr + offset, config.minSilenceDuration || 0.3, 'float');
      offset += 4;
      
      Module.setValue(ptr + offset, config.minSpeechDuration || 0.1, 'float');
      offset += 4;
      
      Module.setValue(ptr + offset, config.windowSize || 512, 'i32');
      offset += 4;
      
      Module.setValue(ptr + offset, config.maxSpeechDuration || 30.0, 'float');
      offset += 4;
      
      return {
        buffer: modelString.ptr, 
        ptr: ptr, 
        len: len
      };
    },
    
    /**
     * Initialize VAD model configuration in WASM
     * @param {Object} config - VAD configuration
     * @param {Object} Module - WebAssembly module
     * @returns {Object} Configuration with WASM pointers
     * @private
     */
    _initVadModelConfig: function(config, Module) {
      if (!('sileroVad' in config)) {
        throw new Error("Missing sileroVad configuration");
      }
      
      const sileroVad = this._initSileroVadConfig(config.sileroVad, Module);
      
      const providerString = SherpaOnnx.Utils.allocateString(config.provider || 'cpu', Module);
      
      const len = sileroVad.len + 4 * 4;
      const ptr = Module._malloc(len);
      
      let offset = 0;
      Module._CopyHeap(sileroVad.ptr, sileroVad.len, ptr + offset);
      offset += sileroVad.len;
      
      Module.setValue(ptr + offset, config.sampleRate || 16000, 'i32');
      offset += 4;
      
      Module.setValue(ptr + offset, config.numThreads || 1, 'i32');
      offset += 4;
      
      Module.setValue(ptr + offset, providerString.ptr, 'i8*');  // provider
      offset += 4;
      
      Module.setValue(ptr + offset, config.debug !== undefined ? config.debug : 1, 'i32');
      offset += 4;
      
      return {
        buffer: providerString.ptr, 
        ptr: ptr, 
        len: len, 
        sileroVad: sileroVad
      };
    }
  };
  
})(typeof window !== 'undefined' ? window : global); 