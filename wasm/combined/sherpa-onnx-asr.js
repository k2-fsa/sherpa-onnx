/**
 * sherpa-onnx-asr.js
 * 
 * Automatic Speech Recognition functionality for SherpaOnnx
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
  
  // Create or use existing ASR namespace
  SherpaOnnx.ASR = SherpaOnnx.ASR || {};
  
  // Define the ASR module functionality
  SherpaOnnx.ASR = {
    /**
     * Load an ASR model from URLs
     * @param {Object} modelConfig - Configuration for the model
     * @returns {Promise<Object>} - Information about the loaded model
     */
    loadModel: async function(modelConfig) {
      const modelDir = modelConfig.modelDir || 'asr-models';
      
      // Create directory if it doesn't exist
      try {
        global.Module.FS.mkdir(modelDir, 0o777);
      } catch(e) {
        if (e.code !== 'EEXIST') throw e;
      }
      
      // Collection for actual file paths
      const actualPaths = {};
      
      // Load model files based on type
      if (modelConfig.type === 'transducer') {
        const results = await Promise.all([
          SherpaOnnx.FileSystem.safeLoadFile(modelConfig.encoder || 'assets/asr/encoder.onnx', `${modelDir}/encoder.onnx`, modelConfig.debug),
          SherpaOnnx.FileSystem.safeLoadFile(modelConfig.decoder || 'assets/asr/decoder.onnx', `${modelDir}/decoder.onnx`, modelConfig.debug),
          SherpaOnnx.FileSystem.safeLoadFile(modelConfig.joiner || 'assets/asr/joiner.onnx', `${modelDir}/joiner.onnx`, modelConfig.debug),
          SherpaOnnx.FileSystem.safeLoadFile(modelConfig.tokens || 'assets/asr/tokens.txt', `${modelDir}/tokens.txt`, modelConfig.debug)
        ]);
        
        // Collect actual paths
        actualPaths.encoder = results[0].path;
        actualPaths.decoder = results[1].path;
        actualPaths.joiner = results[2].path;
        actualPaths.tokens = results[3].path;
        
      } else if (modelConfig.type === 'paraformer') {
        const results = await Promise.all([
          SherpaOnnx.FileSystem.safeLoadFile(modelConfig.encoder || 'assets/asr/encoder.onnx', `${modelDir}/encoder.onnx`, modelConfig.debug),
          SherpaOnnx.FileSystem.safeLoadFile(modelConfig.decoder || 'assets/asr/decoder.onnx', `${modelDir}/decoder.onnx`, modelConfig.debug),
          SherpaOnnx.FileSystem.safeLoadFile(modelConfig.tokens || 'assets/asr/tokens.txt', `${modelDir}/tokens.txt`, modelConfig.debug)
        ]);
        
        // Collect actual paths
        actualPaths.encoder = results[0].path;
        actualPaths.decoder = results[1].path;
        actualPaths.tokens = results[2].path;
        
      } else if (modelConfig.type === 'ctc') {
        const results = await Promise.all([
          SherpaOnnx.FileSystem.safeLoadFile(modelConfig.model || 'assets/asr/model.onnx', `${modelDir}/model.onnx`, modelConfig.debug),
          SherpaOnnx.FileSystem.safeLoadFile(modelConfig.tokens || 'assets/asr/tokens.txt', `${modelDir}/tokens.txt`, modelConfig.debug)
        ]);
        
        // Collect actual paths
        actualPaths.model = results[0].path;
        actualPaths.tokens = results[1].path;
      }
      
      // Get base directory from the tokens path
      let effectiveModelDir = modelDir;
      if (actualPaths.tokens) {
        const lastSlash = actualPaths.tokens.lastIndexOf('/');
        if (lastSlash > 0) {
          effectiveModelDir = actualPaths.tokens.substring(0, lastSlash);
        }
      }
      
      return {
        modelDir: effectiveModelDir,
        type: modelConfig.type,
        actualPaths: actualPaths
      };
    },
    
    /**
     * Initialize online recognizer configuration in WASM
     * @param {Object} config - ASR configuration
     * @param {Object} Module - WebAssembly module
     * @returns {number} - Pointer to the configuration in WASM
     * @private
     */
    _initOnlineRecognizerConfig: function(config, Module) {
      if (!config) {
        console.error('ASR config is null');
        return 0;
      }

      try {
        // First, allocate all the strings we need
        const allocatedStrings = {};
        
        // Transducer model config
        if (config.modelConfig.transducer) {
          allocatedStrings.encoder = SherpaOnnx.Utils.allocateString(config.modelConfig.transducer.encoder, Module);
          allocatedStrings.decoder = SherpaOnnx.Utils.allocateString(config.modelConfig.transducer.decoder, Module);
          allocatedStrings.joiner = SherpaOnnx.Utils.allocateString(config.modelConfig.transducer.joiner, Module);
        } else {
          allocatedStrings.encoder = SherpaOnnx.Utils.allocateString('', Module);
          allocatedStrings.decoder = SherpaOnnx.Utils.allocateString('', Module);
          allocatedStrings.joiner = SherpaOnnx.Utils.allocateString('', Module);
        }
        
        // Paraformer model config
        if (config.modelConfig.paraformer) {
          allocatedStrings.paraEncoder = SherpaOnnx.Utils.allocateString(config.modelConfig.paraformer.encoder, Module);
          allocatedStrings.paraDecoder = SherpaOnnx.Utils.allocateString(config.modelConfig.paraformer.decoder, Module);
        } else {
          allocatedStrings.paraEncoder = SherpaOnnx.Utils.allocateString('', Module);
          allocatedStrings.paraDecoder = SherpaOnnx.Utils.allocateString('', Module);
        }
        
        // Zipformer2 CTC model config
        if (config.modelConfig.zipformer2Ctc) {
          allocatedStrings.zipformerModel = SherpaOnnx.Utils.allocateString(config.modelConfig.zipformer2Ctc.model, Module);
        } else {
          allocatedStrings.zipformerModel = SherpaOnnx.Utils.allocateString('', Module);
        }
        
        // Tokens, provider, model_type, modeling_unit, bpe_vocab
        allocatedStrings.tokens = SherpaOnnx.Utils.allocateString(config.modelConfig.tokens, Module);
        allocatedStrings.provider = SherpaOnnx.Utils.allocateString(config.modelConfig.provider || 'cpu', Module);
        allocatedStrings.modelType = SherpaOnnx.Utils.allocateString('', Module); // Not used in JS API
        allocatedStrings.modelingUnit = SherpaOnnx.Utils.allocateString('', Module); // Not used in JS API
        allocatedStrings.bpeVocab = SherpaOnnx.Utils.allocateString('', Module); // Not used in JS API
        
        // Token buffer is not used in JS API
        allocatedStrings.tokensBuffer = SherpaOnnx.Utils.allocateString('', Module);
        
        // Decoding method
        allocatedStrings.decodingMethod = SherpaOnnx.Utils.allocateString(config.decodingMethod || 'greedy_search', Module);
        
        // Hotwords
        allocatedStrings.hotwordsFile = SherpaOnnx.Utils.allocateString('', Module); // Not used in JS API
        allocatedStrings.hotwordsBuffer = SherpaOnnx.Utils.allocateString('', Module); // Not used in JS API
        
        // Rule FSTs and FARs
        allocatedStrings.ruleFsts = SherpaOnnx.Utils.allocateString('', Module); // Not used in JS API
        allocatedStrings.ruleFars = SherpaOnnx.Utils.allocateString('', Module); // Not used in JS API
        
        // Now allocate the main config structure
        // Size needs to match the C structure size
        const configSize = 200; // Adjust if needed to match C struct
        const configPtr = Module._malloc(configSize);
        
        // Zero out the memory
        Module.HEAP8.fill(0, configPtr, configPtr + configSize);
        
        // Set feat_config fields
        let offset = 0;
        Module.setValue(configPtr + offset, config.featConfig.sampleRate || 16000, 'i32');
        offset += 4;
        Module.setValue(configPtr + offset, config.featConfig.featureDim || 80, 'i32');
        offset += 4;
        
        // Set model_config fields - transducer
        Module.setValue(configPtr + offset, allocatedStrings.encoder.ptr, 'i8*');
        offset += 4;
        Module.setValue(configPtr + offset, allocatedStrings.decoder.ptr, 'i8*');
        offset += 4;
        Module.setValue(configPtr + offset, allocatedStrings.joiner.ptr, 'i8*');
        offset += 4;
        
        // Set model_config fields - paraformer
        Module.setValue(configPtr + offset, allocatedStrings.paraEncoder.ptr, 'i8*');
        offset += 4;
        Module.setValue(configPtr + offset, allocatedStrings.paraDecoder.ptr, 'i8*');
        offset += 4;
        
        // Set model_config fields - zipformer2_ctc
        Module.setValue(configPtr + offset, allocatedStrings.zipformerModel.ptr, 'i8*');
        offset += 4;
        
        // Set remaining model_config fields
        Module.setValue(configPtr + offset, allocatedStrings.tokens.ptr, 'i8*');
        offset += 4;
        Module.setValue(configPtr + offset, config.modelConfig.numThreads || 1, 'i32');
        offset += 4;
        Module.setValue(configPtr + offset, allocatedStrings.provider.ptr, 'i8*');
        offset += 4;
        Module.setValue(configPtr + offset, config.modelConfig.debug || 0, 'i32');
        offset += 4;
        Module.setValue(configPtr + offset, allocatedStrings.modelType.ptr, 'i8*');
        offset += 4;
        Module.setValue(configPtr + offset, allocatedStrings.modelingUnit.ptr, 'i8*');
        offset += 4;
        Module.setValue(configPtr + offset, allocatedStrings.bpeVocab.ptr, 'i8*');
        offset += 4;
        Module.setValue(configPtr + offset, allocatedStrings.tokensBuffer.ptr, 'i8*');
        offset += 4;
        Module.setValue(configPtr + offset, 0, 'i32'); // tokens_buf_size
        offset += 4;
        
        // Set recognizer config fields
        Module.setValue(configPtr + offset, allocatedStrings.decodingMethod.ptr, 'i8*');
        offset += 4;
        Module.setValue(configPtr + offset, config.maxActivePaths || 4, 'i32');
        offset += 4;
        Module.setValue(configPtr + offset, config.enableEndpoint || 1, 'i32');
        offset += 4;
        Module.setValue(configPtr + offset, config.rule1MinTrailingSilence || 2.4, 'float');
        offset += 4;
        Module.setValue(configPtr + offset, config.rule2MinTrailingSilence || 1.2, 'float');
        offset += 4;
        Module.setValue(configPtr + offset, config.rule3MinUtteranceLength || 300, 'float');
        offset += 4;
        
        // Set hotwords fields
        Module.setValue(configPtr + offset, allocatedStrings.hotwordsFile.ptr, 'i8*');
        offset += 4;
        Module.setValue(configPtr + offset, 0.0, 'float'); // hotwords_score
        offset += 4;
        
        // Set CTC FST decoder config - graph and max_active
        Module.setValue(configPtr + offset, 0, 'i8*'); // graph
        offset += 4;
        Module.setValue(configPtr + offset, 0, 'i32'); // max_active
        offset += 4;
        
        // Set rule FSTs and FARs
        Module.setValue(configPtr + offset, allocatedStrings.ruleFsts.ptr, 'i8*');
        offset += 4;
        Module.setValue(configPtr + offset, allocatedStrings.ruleFars.ptr, 'i8*');
        offset += 4;
        
        // Set blank penalty
        Module.setValue(configPtr + offset, 0.0, 'float'); // blank_penalty
        offset += 4;
        
        // Set hotwords buffer and size
        Module.setValue(configPtr + offset, allocatedStrings.hotwordsBuffer.ptr, 'i8*');
        offset += 4;
        Module.setValue(configPtr + offset, 0, 'i32'); // hotwords_buf_size
        offset += 4;
        
        // Save the allocated strings for freeing later
        Module.SherpaOnnxAllocatedStrings = allocatedStrings;
        
        return configPtr;
      } catch (error) {
        console.error('Error initializing ASR config:', error);
        return 0;
      }
    },
    
    /**
     * Free the configuration memory
     * @param {number} configPtr - Pointer to the configuration
     * @param {Object} Module - WebAssembly module
     * @private
     */
    _freeConfig: function(configPtr, Module) {
      if (!configPtr) return;
      
      try {
        // Free all allocated strings
        if (Module.SherpaOnnxAllocatedStrings) {
          for (const key in Module.SherpaOnnxAllocatedStrings) {
            if (Module.SherpaOnnxAllocatedStrings[key].ptr) {
              Module._free(Module.SherpaOnnxAllocatedStrings[key].ptr);
            }
          }
          delete Module.SherpaOnnxAllocatedStrings;
        }
        
        // Free the config structure itself
        Module._free(configPtr);
      } catch (error) {
        console.error('Error freeing ASR config:', error);
      }
    },
    
    /**
     * Create an online ASR recognizer with the loaded model
     * @param {Object} loadedModel - Model information returned by loadModel
     * @param {Object} options - Additional configuration options
     * @returns {OnlineRecognizer} - An instance of OnlineRecognizer
     */
    createOnlineRecognizer: function(loadedModel, options = {}) {
      const config = {
        featConfig: {
          sampleRate: options.sampleRate || 16000,
          featureDim: options.featureDim || 80,
        },
        modelConfig: {
          tokens: loadedModel.actualPaths.tokens || `${loadedModel.modelDir}/tokens.txt`,
          numThreads: options.numThreads || 1,
          provider: options.provider || 'cpu',
          debug: options.debug !== undefined ? options.debug : 1, // Configurable debug
        },
        decodingMethod: options.decodingMethod || 'greedy_search',
        enableEndpoint: options.enableEndpoint === undefined ? 1 : options.enableEndpoint,
        maxActivePaths: options.maxActivePaths || 4,
        rule1MinTrailingSilence: options.rule1MinTrailingSilence || 2.4,
        rule2MinTrailingSilence: options.rule2MinTrailingSilence || 1.2,
        rule3MinUtteranceLength: options.rule3MinUtteranceLength || 300.0,
      };
      
      if (loadedModel.type === 'transducer') {
        config.modelConfig.transducer = {
          encoder: loadedModel.actualPaths.encoder || `${loadedModel.modelDir}/encoder.onnx`,
          decoder: loadedModel.actualPaths.decoder || `${loadedModel.modelDir}/decoder.onnx`,
          joiner: loadedModel.actualPaths.joiner || `${loadedModel.modelDir}/joiner.onnx`,
        };
      } else if (loadedModel.type === 'paraformer') {
        config.modelConfig.paraformer = {
          encoder: loadedModel.actualPaths.encoder || `${loadedModel.modelDir}/encoder.onnx`,
          decoder: loadedModel.actualPaths.decoder || `${loadedModel.modelDir}/decoder.onnx`,
        };
      } else if (loadedModel.type === 'ctc') {
        config.modelConfig.zipformer2Ctc = {
          model: loadedModel.actualPaths.model || `${loadedModel.modelDir}/model.onnx`,
        };
      }
      
      const recognizer = new global.OnlineRecognizer(config, global.Module);
      
      // Track the resource for cleanup if tracking function is available
      if (SherpaOnnx.trackResource) {
        SherpaOnnx.trackResource('asr', recognizer);
      }
      
      return recognizer;
    },
    
    /**
     * Create an offline ASR recognizer with the loaded model
     * @param {Object} loadedModel - Model information returned by loadModel
     * @param {Object} options - Additional configuration options
     * @returns {OfflineRecognizer} - An instance of OfflineRecognizer
     */
    createOfflineRecognizer: function(loadedModel, options = {}) {
      const config = {
        featConfig: {
          sampleRate: options.sampleRate || 16000,
          featureDim: options.featureDim || 80,
        },
        modelConfig: {
          tokens: loadedModel.actualPaths.tokens || `${loadedModel.modelDir}/tokens.txt`,
          numThreads: options.numThreads || 1,
          provider: options.provider || 'cpu',
          debug: options.debug !== undefined ? options.debug : 1, // Configurable debug
        },
        lmConfig: {
          model: '', // No language model by default
          scale: 1.0,
        },
        decodingMethod: options.decodingMethod || 'greedy_search',
        maxActivePaths: options.maxActivePaths || 4,
      };
      
      if (loadedModel.type === 'transducer') {
        config.modelConfig.transducer = {
          encoder: loadedModel.actualPaths.encoder || `${loadedModel.modelDir}/encoder.onnx`,
          decoder: loadedModel.actualPaths.decoder || `${loadedModel.modelDir}/decoder.onnx`,
          joiner: loadedModel.actualPaths.joiner || `${loadedModel.modelDir}/joiner.onnx`,
        };
      } else if (loadedModel.type === 'paraformer') {
        config.modelConfig.paraformer = {
          model: loadedModel.actualPaths.model || `${loadedModel.modelDir}/model.onnx`,
        };
      } else if (loadedModel.type === 'ctc') {
        config.modelConfig.nemoCtc = {
          model: loadedModel.actualPaths.model || `${loadedModel.modelDir}/model.onnx`,
        };
      }
      
      const recognizer = new global.OfflineRecognizer(config, global.Module);
      
      // Track the resource for cleanup if tracking function is available
      if (SherpaOnnx.trackResource) {
        SherpaOnnx.trackResource('asr', recognizer);
      }
      
      return recognizer;
    },
    
    /**
     * Initialize offline recognizer configuration in WASM
     * @param {Object} config - ASR configuration
     * @param {Object} Module - WebAssembly module
     * @returns {number} - Pointer to the configuration in WASM
     * @private
     */
    _initOfflineRecognizerConfig: function(config, Module) {
      if (!config) {
        console.error('ASR config is null');
        return 0;
      }

      try {
        // First, allocate all the strings we need
        const allocatedStrings = {};
        
        // Transducer model config
        if (config.modelConfig.transducer) {
          allocatedStrings.encoder = SherpaOnnx.Utils.allocateString(config.modelConfig.transducer.encoder, Module);
          allocatedStrings.decoder = SherpaOnnx.Utils.allocateString(config.modelConfig.transducer.decoder, Module);
          allocatedStrings.joiner = SherpaOnnx.Utils.allocateString(config.modelConfig.transducer.joiner, Module);
        } else {
          allocatedStrings.encoder = SherpaOnnx.Utils.allocateString('', Module);
          allocatedStrings.decoder = SherpaOnnx.Utils.allocateString('', Module);
          allocatedStrings.joiner = SherpaOnnx.Utils.allocateString('', Module);
        }
        
        // Paraformer model config
        if (config.modelConfig.paraformer) {
          allocatedStrings.paraEncoder = SherpaOnnx.Utils.allocateString(config.modelConfig.paraformer.encoder, Module);
          allocatedStrings.paraDecoder = SherpaOnnx.Utils.allocateString(config.modelConfig.paraformer.decoder, Module);
        } else {
          allocatedStrings.paraEncoder = SherpaOnnx.Utils.allocateString('', Module);
          allocatedStrings.paraDecoder = SherpaOnnx.Utils.allocateString('', Module);
        }
        
        // Zipformer2 CTC model config
        if (config.modelConfig.zipformer2Ctc) {
          allocatedStrings.zipformerModel = SherpaOnnx.Utils.allocateString(config.modelConfig.zipformer2Ctc.model, Module);
        } else {
          allocatedStrings.zipformerModel = SherpaOnnx.Utils.allocateString('', Module);
        }
        
        // Tokens, provider, model_type, modeling_unit, bpe_vocab
        allocatedStrings.tokens = SherpaOnnx.Utils.allocateString(config.modelConfig.tokens, Module);
        allocatedStrings.provider = SherpaOnnx.Utils.allocateString(config.modelConfig.provider || 'cpu', Module);
        allocatedStrings.modelType = SherpaOnnx.Utils.allocateString('', Module); // Not used in JS API
        allocatedStrings.modelingUnit = SherpaOnnx.Utils.allocateString('', Module); // Not used in JS API
        allocatedStrings.bpeVocab = SherpaOnnx.Utils.allocateString('', Module); // Not used in JS API
        
        // Token buffer is not used in JS API
        allocatedStrings.tokensBuffer = SherpaOnnx.Utils.allocateString('', Module);
        
        // Decoding method
        allocatedStrings.decodingMethod = SherpaOnnx.Utils.allocateString(config.decodingMethod || 'greedy_search', Module);
        
        // Hotwords
        allocatedStrings.hotwordsFile = SherpaOnnx.Utils.allocateString('', Module); // Not used in JS API
        allocatedStrings.hotwordsBuffer = SherpaOnnx.Utils.allocateString('', Module); // Not used in JS API
        
        // Rule FSTs and FARs
        allocatedStrings.ruleFsts = SherpaOnnx.Utils.allocateString('', Module); // Not used in JS API
        allocatedStrings.ruleFars = SherpaOnnx.Utils.allocateString('', Module); // Not used in JS API
        
        // Now allocate the main config structure
        // Size needs to match the C structure size
        const configSize = 200; // Adjust if needed to match C struct
        const configPtr = Module._malloc(configSize);
        
        // Zero out the memory
        Module.HEAP8.fill(0, configPtr, configPtr + configSize);
        
        // Set feat_config fields
        let offset = 0;
        Module.setValue(configPtr + offset, config.featConfig.sampleRate || 16000, 'i32');
        offset += 4;
        Module.setValue(configPtr + offset, config.featConfig.featureDim || 80, 'i32');
        offset += 4;
        
        // Set model_config fields - transducer
        Module.setValue(configPtr + offset, allocatedStrings.encoder.ptr, 'i8*');
        offset += 4;
        Module.setValue(configPtr + offset, allocatedStrings.decoder.ptr, 'i8*');
        offset += 4;
        Module.setValue(configPtr + offset, allocatedStrings.joiner.ptr, 'i8*');
        offset += 4;
        
        // Set model_config fields - paraformer
        Module.setValue(configPtr + offset, allocatedStrings.paraEncoder.ptr, 'i8*');
        offset += 4;
        Module.setValue(configPtr + offset, allocatedStrings.paraDecoder.ptr, 'i8*');
        offset += 4;
        
        // Set model_config fields - zipformer2_ctc
        Module.setValue(configPtr + offset, allocatedStrings.zipformerModel.ptr, 'i8*');
        offset += 4;
        
        // Set remaining model_config fields
        Module.setValue(configPtr + offset, allocatedStrings.tokens.ptr, 'i8*');
        offset += 4;
        Module.setValue(configPtr + offset, config.modelConfig.numThreads || 1, 'i32');
        offset += 4;
        Module.setValue(configPtr + offset, allocatedStrings.provider.ptr, 'i8*');
        offset += 4;
        Module.setValue(configPtr + offset, config.modelConfig.debug || 0, 'i32');
        offset += 4;
        Module.setValue(configPtr + offset, allocatedStrings.modelType.ptr, 'i8*');
        offset += 4;
        Module.setValue(configPtr + offset, allocatedStrings.modelingUnit.ptr, 'i8*');
        offset += 4;
        Module.setValue(configPtr + offset, allocatedStrings.bpeVocab.ptr, 'i8*');
        offset += 4;
        Module.setValue(configPtr + offset, allocatedStrings.tokensBuffer.ptr, 'i8*');
        offset += 4;
        Module.setValue(configPtr + offset, 0, 'i32'); // tokens_buf_size
        offset += 4;
        
        // Set recognizer config fields
        Module.setValue(configPtr + offset, allocatedStrings.decodingMethod.ptr, 'i8*');
        offset += 4;
        Module.setValue(configPtr + offset, config.maxActivePaths || 4, 'i32');
        offset += 4;
        Module.setValue(configPtr + offset, config.enableEndpoint || 1, 'i32');
        offset += 4;
        Module.setValue(configPtr + offset, config.rule1MinTrailingSilence || 2.4, 'float');
        offset += 4;
        Module.setValue(configPtr + offset, config.rule2MinTrailingSilence || 1.2, 'float');
        offset += 4;
        Module.setValue(configPtr + offset, config.rule3MinUtteranceLength || 300, 'float');
        offset += 4;
        
        // Set hotwords fields
        Module.setValue(configPtr + offset, allocatedStrings.hotwordsFile.ptr, 'i8*');
        offset += 4;
        Module.setValue(configPtr + offset, 0.0, 'float'); // hotwords_score
        offset += 4;
        
        // Set CTC FST decoder config - graph and max_active
        Module.setValue(configPtr + offset, 0, 'i8*'); // graph
        offset += 4;
        Module.setValue(configPtr + offset, 0, 'i32'); // max_active
        offset += 4;
        
        // Set rule FSTs and FARs
        Module.setValue(configPtr + offset, allocatedStrings.ruleFsts.ptr, 'i8*');
        offset += 4;
        Module.setValue(configPtr + offset, allocatedStrings.ruleFars.ptr, 'i8*');
        offset += 4;
        
        // Set blank penalty
        Module.setValue(configPtr + offset, 0.0, 'float'); // blank_penalty
        offset += 4;
        
        // Set hotwords buffer and size
        Module.setValue(configPtr + offset, allocatedStrings.hotwordsBuffer.ptr, 'i8*');
        offset += 4;
        Module.setValue(configPtr + offset, 0, 'i32'); // hotwords_buf_size
        offset += 4;
        
        // Save the allocated strings for freeing later
        Module.SherpaOnnxAllocatedStrings = allocatedStrings;
        
        return configPtr;
      } catch (error) {
        console.error('Error initializing ASR config:', error);
        return 0;
      }
    }
  };
  
  /**
   * OnlineRecognizer class for streaming speech recognition
   */
  global.OnlineRecognizer = global.OnlineRecognizer || function(config, Module) {
    this.Module = Module;
    this.config = config;
    this.streams = []; // Track streams created by this recognizer
    
    // Initialize the configuration in WASM
    const configPtr = SherpaOnnx.ASR._initOnlineRecognizerConfig(config, Module);
    
    // Create the recognizer
    this.handle = Module.ccall(
      'SherpaOnnxCreateOnlineRecognizer',
      'number',
      ['number'],
      [configPtr]
    );
    
    // Free the configuration memory
    SherpaOnnx.ASR._freeConfig(configPtr, Module);
    
    /**
     * Create a stream for audio input
     * @returns {OnlineStream} - A new stream for audio input
     */
    this.createStream = function() {
      const streamHandle = this.Module.ccall(
        'SherpaOnnxCreateOnlineStream',
        'number',
        ['number'],
        [this.handle]
      );
      const stream = new global.OnlineStream(streamHandle, this.Module);
      
      // Track the stream for cleanup
      this.streams.push(stream);
      
      return stream;
    };
    
    /**
     * Check if the stream is ready for decoding
     * @param {OnlineStream} stream - The stream to check
     * @returns {boolean} - True if ready, false otherwise
     */
    this.isReady = function(stream) {
      return this.Module.ccall(
        'SherpaOnnxIsOnlineStreamReady',
        'number',
        ['number', 'number'],
        [this.handle, stream.handle]
      ) === 1;
    };
    
    /**
     * Decode the audio in the stream
     * @param {OnlineStream} stream - The stream to decode
     */
    this.decode = function(stream) {
      this.Module.ccall(
        'SherpaOnnxDecodeOnlineStream',
        'void',
        ['number', 'number'],
        [this.handle, stream.handle]
      );
    };
    
    /**
     * Check if an endpoint has been detected
     * @param {OnlineStream} stream - The stream to check
     * @returns {boolean} - True if endpoint detected, false otherwise
     */
    this.isEndpoint = function(stream) {
      return this.Module.ccall(
        'SherpaOnnxOnlineStreamIsEndpoint',
        'number',
        ['number', 'number'],
        [this.handle, stream.handle]
      ) === 1;
    };
    
    /**
     * Reset the stream
     * @param {OnlineStream} stream - The stream to reset
     */
    this.reset = function(stream) {
      this.Module.ccall(
        'SherpaOnnxOnlineStreamReset',
        'void',
        ['number', 'number'],
        [this.handle, stream.handle]
      );
    };
    
    /**
     * Get the recognition result
     * @param {OnlineStream} stream - The stream to get results from
     * @returns {Object} - Recognition result as JSON
     */
    this.getResult = function(stream) {
      const resultPtr = this.Module.ccall(
        'SherpaOnnxGetOnlineStreamResultAsJson',
        'number',
        ['number', 'number'],
        [this.handle, stream.handle]
      );
      
      const jsonStr = this.Module.UTF8ToString(resultPtr);
      const result = JSON.parse(jsonStr);
      
      this.Module.ccall(
        'SherpaOnnxDestroyOnlineStreamResultJson',
        'null',
        ['number'],
        [resultPtr]
      );
      
      return result;
    };
    
    /**
     * Free the recognizer and all associated streams
     */
    this.free = function() {
      // Free all streams first
      for (let i = this.streams.length - 1; i >= 0; i--) {
        if (this.streams[i]) {
          this.streams[i].free();
        }
        this.streams.splice(i, 1);
      }
      
      // Then free the recognizer
      if (this.handle) {
        this.Module.ccall(
          'SherpaOnnxDestroyOnlineRecognizer',
          'null',
          ['number'],
          [this.handle]
        );
        this.handle = null;
      }
    };
  };
  
  /**
   * OnlineStream class for handling streaming audio input
   */
  global.OnlineStream = global.OnlineStream || function(handle, Module) {
    this.handle = handle;
    this.Module = Module;
    this.pointer = null;  // buffer
    this.n = 0;           // buffer size
    
    /**
     * Accept audio waveform data
     * @param {number} sampleRate - Sample rate of the audio
     * @param {Float32Array} samples - Audio samples in [-1, 1] range
     */
    this.acceptWaveform = function(sampleRate, samples) {
      if (this.n < samples.length) {
        if (this.pointer) {
          this.Module._free(this.pointer);
        }
        this.pointer = this.Module._malloc(samples.length * samples.BYTES_PER_ELEMENT);
        this.n = samples.length;
      }

      this.Module.HEAPF32.set(samples, this.pointer / samples.BYTES_PER_ELEMENT);
      this.Module.ccall(
        'SherpaOnnxOnlineStreamAcceptWaveform',
        'void',
        ['number', 'number', 'number', 'number'],
        [this.handle, sampleRate, this.pointer, samples.length]
      );
    };
    
    /**
     * Signal that input is finished
     */
    this.inputFinished = function() {
      this.Module.ccall(
        'SherpaOnnxOnlineStreamInputFinished',
        'void',
        ['number'],
        [this.handle]
      );
    };
    
    /**
     * Free the stream
     */
    this.free = function() {
      if (this.handle) {
        this.Module.ccall(
          'SherpaOnnxDestroyOnlineStream',
          'null',
          ['number'],
          [this.handle]
        );
        this.handle = null;
        
        if (this.pointer) {
          this.Module._free(this.pointer);
          this.pointer = null;
          this.n = 0;
        }
      }
    };
  };
  
  /**
   * OfflineRecognizer class for non-streaming speech recognition
   */
  global.OfflineRecognizer = global.OfflineRecognizer || function(config, Module) {
    this.Module = Module;
    this.config = config;
    this.streams = []; // Track streams created by this recognizer
    
    // Initialize the configuration in WASM
    const configPtr = SherpaOnnx.ASR._initOfflineRecognizerConfig(config, Module);
    
    // Create the recognizer
    this.handle = Module.ccall(
      'SherpaOnnxCreateOfflineRecognizer',
      'number',
      ['number'],
      [configPtr]
    );
    
    // Free the configuration memory
    SherpaOnnx.ASR._freeConfig(configPtr, Module);
    
    /**
     * Create a stream for offline processing
     * @returns {OfflineStream} - A new stream for offline processing
     */
    this.createStream = function() {
      const streamHandle = this.Module.ccall(
        'SherpaOnnxCreateOfflineStream',
        'number',
        ['number'],
        [this.handle]
      );
      const stream = new global.OfflineStream(streamHandle, this.Module);
      
      // Track the stream for cleanup
      this.streams.push(stream);
      
      return stream;
    };
    
    /**
     * Decode the audio in the stream
     * @param {OfflineStream} stream - The stream to decode
     */
    this.decode = function(stream) {
      this.Module.ccall(
        'SherpaOnnxDecodeOfflineStream',
        'void',
        ['number', 'number'],
        [this.handle, stream.handle]
      );
    };
    
    /**
     * Free the recognizer and all associated streams
     */
    this.free = function() {
      // Free all streams first
      for (let i = this.streams.length - 1; i >= 0; i--) {
        if (this.streams[i]) {
          this.streams[i].free();
        }
        this.streams.splice(i, 1);
      }
      
      // Then free the recognizer
      if (this.handle) {
        this.Module.ccall(
          'SherpaOnnxDestroyOfflineRecognizer',
          'null',
          ['number'],
          [this.handle]
        );
        this.handle = null;
      }
    };
  };
  
  /**
   * OfflineStream class for handling non-streaming audio input
   */
  global.OfflineStream = global.OfflineStream || function(handle, Module) {
    this.handle = handle;
    this.Module = Module;
    
    /**
     * Accept audio waveform data
     * @param {number} sampleRate - Sample rate of the audio
     * @param {Float32Array} samples - Audio samples in [-1, 1] range
     */
    this.acceptWaveform = function(sampleRate, samples) {
      const pointer = this.Module._malloc(samples.length * samples.BYTES_PER_ELEMENT);
      this.Module.HEAPF32.set(samples, pointer / samples.BYTES_PER_ELEMENT);
      
      this.Module.ccall(
        'SherpaOnnxAcceptWaveformOffline',
        'void',
        ['number', 'number', 'number', 'number'],
        [this.handle, sampleRate, pointer, samples.length]
      );
      
      this.Module._free(pointer);
    };
    
    /**
     * Get the recognition result
     * @returns {Object} - Recognition result as JSON
     */
    this.getResult = function() {
      const resultPtr = this.Module.ccall(
        'SherpaOnnxGetOfflineStreamResultAsJson',
        'number',
        ['number'],
        [this.handle]
      );
      
      const jsonStr = this.Module.UTF8ToString(resultPtr);
      const result = JSON.parse(jsonStr);
      
      this.Module.ccall(
        'SherpaOnnxDestroyOfflineStreamResultJson',
        'null',
        ['number'],
        [resultPtr]
      );
      
      return result;
    };
    
    /**
     * Free the stream
     */
    this.free = function() {
      if (this.handle) {
        this.Module.ccall(
          'SherpaOnnxDestroyOfflineStream',
          'null',
          ['number'],
          [this.handle]
        );
        this.handle = null;
      }
    };
  };
  
  // For Node.js environments
  if (typeof module !== 'undefined' && module.exports) {
    module.exports = SherpaOnnx;
  }
})(typeof window !== 'undefined' ? window : global); 