/**
 * sherpa-onnx-kws.js
 * 
 * Keyword Spotting functionality for SherpaOnnx
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
  
  // Create or use existing KWS namespace
  SherpaOnnx.KWS = SherpaOnnx.KWS || {};
  
  // Define the KWS module functionality
  SherpaOnnx.KWS = {
    /**
     * Load a KWS model from URLs
     * @param {Object} modelConfig - Configuration for the model
     * @returns {Promise<Object>} - Information about the loaded model
     */
    loadModel: async function(modelConfig) {
      const modelDir = modelConfig.modelDir || 'kws-models';
      const debug = modelConfig.debug || false;
      
      try {
        global.Module.FS.mkdir(modelDir, 0o777);
      } catch(e) {
        if (e.code !== 'EEXIST') throw e;
      }
      
      if (debug) console.log(`Loading KWS model files to ${modelDir}`);
      
      // Load model files and store the actual paths
      const actualPaths = {};
      
      // Load encoder
      const encoderResult = await SherpaOnnx.FileSystem.safeLoadFile(
        modelConfig.encoder || 'assets/kws/encoder.onnx', 
        `${modelDir}/encoder.onnx`, 
        debug
      );
      actualPaths.encoder = encoderResult.path || `${modelDir}/encoder.onnx`;
      if (debug) console.log(`Loaded encoder to ${actualPaths.encoder}`);
      
      // Load decoder
      const decoderResult = await SherpaOnnx.FileSystem.safeLoadFile(
        modelConfig.decoder || 'assets/kws/decoder.onnx', 
        `${modelDir}/decoder.onnx`, 
        debug
      );
      actualPaths.decoder = decoderResult.path || `${modelDir}/decoder.onnx`;
      if (debug) console.log(`Loaded decoder to ${actualPaths.decoder}`);
      
      // Load joiner
      const joinerResult = await SherpaOnnx.FileSystem.safeLoadFile(
        modelConfig.joiner || 'assets/kws/joiner.onnx', 
        `${modelDir}/joiner.onnx`, 
        debug
      );
      actualPaths.joiner = joinerResult.path || `${modelDir}/joiner.onnx`;
      if (debug) console.log(`Loaded joiner to ${actualPaths.joiner}`);
      
      // Load tokens file
      const tokensResult = await SherpaOnnx.FileSystem.safeLoadFile(
        modelConfig.tokens || 'assets/kws/tokens.txt', 
        `${modelDir}/tokens.txt`, 
        debug
      );
      actualPaths.tokens = tokensResult.path || `${modelDir}/tokens.txt`;
      if (debug) console.log(`Loaded tokens to ${actualPaths.tokens}`);
      
      // Load the tokens content for validation
      try {
        const tokensContent = global.Module.FS.readFile(actualPaths.tokens, { encoding: 'utf8' });
        actualPaths.tokensMap = this.parseTokensFile(tokensContent);
        if (debug) console.log(`Parsed ${Object.keys(actualPaths.tokensMap).length} tokens`);
      } catch (e) {
        console.error(`Failed to read tokens file: ${e.message}`);
        actualPaths.tokensMap = null;
      }
      
      // Load keywords file if provided
      if (modelConfig.keywordsFile) {
        const keywordsResult = await SherpaOnnx.FileSystem.safeLoadFile(
          modelConfig.keywordsFile, 
          `${modelDir}/keywords.txt`, 
          debug
        );
        actualPaths.keywordsFile = keywordsResult.path || `${modelDir}/keywords.txt`;
        if (debug) console.log(`Loaded keywords file to ${actualPaths.keywordsFile}`);
      }
      
      return {
        modelDir: modelDir,
        paths: actualPaths
      };
    },
    
    /**
     * Parse the tokens file to create a map of valid tokens
     * @param {string} content - The content of the tokens file
     * @returns {Object} - Map of tokens to their IDs
     */
    parseTokensFile: function(content) {
      const tokensMap = {};
      const lines = content.split('\n');
      
      for (const line of lines) {
        const parts = line.trim().split(' ');
        if (parts.length >= 2) {
          const token = parts[0];
          const id = parseInt(parts[1]);
          if (!isNaN(id)) {
            tokensMap[token] = id;
          }
        }
      }
      
      return tokensMap;
    },
    
    /**
     * Validate keywords against available tokens
     * @param {string} keywords - The keywords to validate
     * @param {Object} tokensMap - Map of valid tokens
     * @returns {Object} - Validation result with formatted keywords
     */
    validateKeywords: function(keywords, tokensMap) {
      if (!tokensMap) return { valid: false, message: 'No tokens available for validation' };
      
      const lines = keywords.split('\n');
      const validatedLines = [];
      const invalidTokens = new Set();
      let isValid = true;
      
      for (const line of lines) {
        // Skip empty lines
        if (!line.trim()) continue;
        
        const parts = line.trim().split('@');
        let phonetic = parts[0].trim();
        const label = parts.length > 1 ? parts[1].trim() : phonetic;
        
        // Validate each token in the phonetic representation
        const tokens = phonetic.split(' ').filter(t => t);
        const validTokens = [];
        
        for (const token of tokens) {
          if (token in tokensMap) {
            validTokens.push(token);
          } else {
            invalidTokens.add(token);
            isValid = false;
          }
        }
        
        const validatedLine = validTokens.join(' ') + ' @' + label;
        validatedLines.push(validatedLine);
      }
      
      return {
        valid: isValid,
        formattedKeywords: validatedLines.join('\n'),
        invalidTokens: [...invalidTokens],
        message: isValid ? 'All keywords are valid' : 
          `Invalid tokens: ${[...invalidTokens].join(', ')}`
      };
    },

    /**
     * Create a Keyword Spotter with the loaded model
     * @param {Object} loadedModel - Model information returned by loadModel
     * @param {Object} options - Additional configuration options
     * @returns {KeywordSpotter} - An instance of KeywordSpotter
     */
    createKeywordSpotter: function(loadedModel, options = {}) {
      const debug = options.debug || false;
      
      // Create transducer configuration using actual paths
      const transducerConfig = {
        encoder: loadedModel.paths.encoder,
        decoder: loadedModel.paths.decoder,
        joiner: loadedModel.paths.joiner,
      };
      
      // Create model configuration
      const modelConfig = {
        transducer: transducerConfig,
        tokens: loadedModel.paths.tokens,
        provider: options.provider || 'cpu',
        numThreads: options.numThreads || 1,
        debug: options.debug !== undefined ? options.debug : 1,
      };
      
      // Create feature configuration
      const featConfig = {
        samplingRate: options.sampleRate || 16000,
        featureDim: options.featureDim || 80,
      };
      
      // First, create a keywords.txt file in the same directory as the tokens file
      const tokensPath = loadedModel.paths.tokens;
      const tokensDir = tokensPath.substring(0, tokensPath.lastIndexOf('/'));
      const keywordsPath = `${tokensDir}/keywords.txt`;
      
      // Default keywords as individual characters
      let defaultKeywords = 
        "h e l l o @Hello\n" + 
        "c o m p u t e r @Computer\n" + 
        "a l e x a @Alexa";
      
      // Use provided keywords or default, then validate
      let keywordsContent = options.keywords || defaultKeywords;
      
      // Validate the keywords against the tokens map if available
      if (loadedModel.paths.tokensMap) {
        const validationResult = this.validateKeywords(keywordsContent, loadedModel.paths.tokensMap);
        
        if (!validationResult.valid) {
          console.warn(`Keyword validation failed: ${validationResult.message}`);
          console.warn('Using only valid tokens for keywords');
        }
        
        keywordsContent = validationResult.formattedKeywords;
        
        if (debug) {
          console.log(`Validation result:`, validationResult);
        }
      }
      
      try {
        // Make sure file exists with absolute path
        global.Module.FS.writeFile(keywordsPath, keywordsContent);
        console.log(`Created keywords file at: ${keywordsPath}`);
        
        if (debug) {
          console.log(`Keywords content: ${keywordsContent}`);
        }
        
        // Verify the file is created
        try {
          const stat = global.Module.FS.stat(keywordsPath);
          if (debug) console.log(`Keywords file exists, size: ${stat.size} bytes`);
        } catch (e) {
          console.error(`Failed to verify keywords file at ${keywordsPath}:`, e);
        }
      } catch (e) {
        console.error('Failed to write keywords file:', e);
      }
      
      // Create the KWS configuration
      const configObj = {
        featConfig: featConfig,
        modelConfig: modelConfig,
        maxActivePaths: options.maxActivePaths || 4,
        numTrailingBlanks: options.numTrailingBlanks || 1,
        keywordsScore: options.keywordsScore || 1.0,
        keywordsThreshold: options.keywordsThreshold || 0.25,
        keywordsFile: keywordsPath
      };
      
      if (debug) {
        console.log('KWS Configuration:', JSON.stringify(configObj, null, 2));
      }
      
      // Create the KWS instance using the global createKws helper
      if (typeof createKws === 'function') {
        return createKws(global.Module, configObj);
      }
      
      // Fall back to our implementation if global function not available
      return new global.Kws(configObj, global.Module);
    }
  };
  
  /**
   * Wrapper for Stream class
   */
  global.Stream = global.Stream || function(handle, Module) {
    this.handle = handle;
    this.Module = Module;
    this.pointer = null;
    this.n = 0;
    
    /**
     * Free the stream
     */
    this.free = function() {
      if (this.handle) {
        this.Module._SherpaOnnxDestroyOnlineStream(this.handle);
        this.handle = null;
        if (this.pointer) {
          this.Module._free(this.pointer);
          this.pointer = null;
          this.n = 0;
        }
      }
    };
    
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
      this.Module._SherpaOnnxOnlineStreamAcceptWaveform(
        this.handle, sampleRate, this.pointer, samples.length);
    };
    
    /**
     * Signal that input is finished
     */
    this.inputFinished = function() {
      this.Module._SherpaOnnxOnlineStreamInputFinished(this.handle);
    };
  };
  
  /**
   * KeywordSpotter class
   */
  global.Kws = global.Kws || function(configObj, Module) {
    this.config = configObj;
    
    // Initialize the configuration
    const config = initKwsConfig(configObj, Module);
    const handle = Module._SherpaOnnxCreateKeywordSpotter(config.ptr);
    
    // Free the configuration
    freeConfig(config, Module);
    
    this.handle = handle;
    this.Module = Module;
    
    /**
     * Free the keyword spotter
     */
    this.free = function() {
      this.Module._SherpaOnnxDestroyKeywordSpotter(this.handle);
      this.handle = 0;
    };
    
    /**
     * Create a stream for keyword spotting
     * @returns {Stream} - A new stream for keyword spotting
     */
    this.createStream = function() {
      const handle = this.Module._SherpaOnnxCreateKeywordStream(this.handle);
      return new global.Stream(handle, this.Module);
    };
    
    /**
     * Check if the stream is ready for decoding
     * @param {Stream} stream - The stream to check
     * @returns {boolean} - True if ready, false otherwise
     */
    this.isReady = function(stream) {
      return this.Module._SherpaOnnxIsKeywordStreamReady(
        this.handle, stream.handle) === 1;
    };
    
    /**
     * Decode the audio in the stream for keyword spotting
     * @param {Stream} stream - The stream to decode
     */
    this.decode = function(stream) {
      this.Module._SherpaOnnxDecodeKeywordStream(this.handle, stream.handle);
    };
    
    /**
     * Reset the stream after keyword detection
     * @param {Stream} stream - The stream to reset
     */
    this.reset = function(stream) {
      this.Module._SherpaOnnxResetKeywordStream(this.handle, stream.handle);
    };
    
    /**
     * Get the keyword spotting result
     * @param {Stream} stream - The stream to get results from
     * @returns {Object} - Keyword spotting result as JSON
     */
    this.getResult = function(stream) {
      const r = this.Module._SherpaOnnxGetKeywordResult(this.handle, stream.handle);
      const jsonPtr = this.Module.getValue(r + 24, 'i8*');
      const json = this.Module.UTF8ToString(jsonPtr);
      this.Module._SherpaOnnxDestroyKeywordResult(r);
      return JSON.parse(json);
    };
  };
  
  // For Node.js environments
  if (typeof module !== 'undefined' && module.exports) {
    module.exports = SherpaOnnx;
  }
})(typeof window !== 'undefined' ? window : global);

/**
 * Initialize feature extractor configuration
 */
function initFeatureExtractorConfig(config, Module) {
  const ptr = Module._malloc(4 * 2);
  Module.setValue(ptr, config.samplingRate || 16000, 'i32');
  Module.setValue(ptr + 4, config.featureDim || 80, 'i32');
  return {
    ptr: ptr, 
    len: 8,
  };
}

/**
 * Initialize transducer model configuration
 */
function initSherpaOnnxOnlineTransducerModelConfig(config, Module) {
  const encoderLen = Module.lengthBytesUTF8(config.encoder) + 1;
  const decoderLen = Module.lengthBytesUTF8(config.decoder) + 1;
  const joinerLen = Module.lengthBytesUTF8(config.joiner) + 1;

  const n = encoderLen + decoderLen + joinerLen;
  const buffer = Module._malloc(n);

  const len = 3 * 4;  // 3 pointers
  const ptr = Module._malloc(len);

  let offset = 0;
  Module.stringToUTF8(config.encoder, buffer + offset, encoderLen);
  offset += encoderLen;

  Module.stringToUTF8(config.decoder, buffer + offset, decoderLen);
  offset += decoderLen;

  Module.stringToUTF8(config.joiner, buffer + offset, joinerLen);

  offset = 0;
  Module.setValue(ptr, buffer + offset, 'i8*');
  offset += encoderLen;

  Module.setValue(ptr + 4, buffer + offset, 'i8*');
  offset += decoderLen;

  Module.setValue(ptr + 8, buffer + offset, 'i8*');

  return {
    buffer: buffer, 
    ptr: ptr, 
    len: len,
  };
}

/**
 * Initialize model configuration
 */
function initModelConfig(config, Module) {
  if (!('tokensBuf' in config)) {
    config.tokensBuf = '';
  }

  if (!('tokensBufSize' in config)) {
    config.tokensBufSize = 0;
  }

  const transducer = initSherpaOnnxOnlineTransducerModelConfig(config.transducer, Module);
  const paraformer_len = 2 * 4;
  const ctc_len = 1 * 4;

  const len = transducer.len + paraformer_len + ctc_len + 9 * 4;
  const ptr = Module._malloc(len);
  Module.HEAPU8.fill(0, ptr, ptr + len);

  let offset = 0;
  Module._CopyHeap(transducer.ptr, transducer.len, ptr + offset);

  const tokensLen = Module.lengthBytesUTF8(config.tokens) + 1;
  const providerLen = Module.lengthBytesUTF8(config.provider || 'cpu') + 1;
  const buffer = Module._malloc(tokensLen + providerLen);

  offset = 0;
  Module.stringToUTF8(config.tokens, buffer, tokensLen);
  offset += tokensLen;

  Module.stringToUTF8(config.provider || 'cpu', buffer + offset, providerLen);

  offset = transducer.len + paraformer_len + ctc_len;
  Module.setValue(ptr + offset, buffer, 'i8*');
  offset += 4;

  Module.setValue(ptr + offset, config.numThreads || 1, 'i32');
  offset += 4;

  Module.setValue(ptr + offset, buffer + tokensLen, 'i8*');
  offset += 4;

  Module.setValue(ptr + offset, config.debug, 'i32');

  return {
    buffer: buffer, 
    ptr: ptr, 
    len: len, 
    transducer: transducer
  };
}

/**
 * Initialize KWS configuration
 */
function initKwsConfig(config, Module) {
  if (!('featConfig' in config)) {
    config.featConfig = {
      samplingRate: 16000,
      featureDim: 80,
    };
  }

  if (!('keywordsBuf' in config)) {
    config.keywordsBuf = '';
  }

  if (!('keywordsBufSize' in config)) {
    config.keywordsBufSize = 0;
  }

  const featConfig = initFeatureExtractorConfig(config.featConfig, Module);
  const modelConfig = initModelConfig(config.modelConfig, Module);
  const numBytes = featConfig.len + modelConfig.len + 4 * 7;

  const ptr = Module._malloc(numBytes);
  let offset = 0;
  Module._CopyHeap(featConfig.ptr, featConfig.len, ptr + offset);
  offset += featConfig.len;

  Module._CopyHeap(modelConfig.ptr, modelConfig.len, ptr + offset);
  offset += modelConfig.len;

  Module.setValue(ptr + offset, config.maxActivePaths || 4, 'i32');
  offset += 4;

  Module.setValue(ptr + offset, config.numTrailingBlanks || 1, 'i32');
  offset += 4;

  Module.setValue(ptr + offset, config.keywordsScore || 1.0, 'float');
  offset += 4;

  Module.setValue(ptr + offset, config.keywordsThreshold || 0.25, 'float');
  offset += 4;

  // Handle keywords file
  let keywordsFileBuffer = 0;
  if (config.keywordsFile) {
    const keywordsFileLen = Module.lengthBytesUTF8(config.keywordsFile) + 1;
    keywordsFileBuffer = Module._malloc(keywordsFileLen);
    Module.stringToUTF8(config.keywordsFile, keywordsFileBuffer, keywordsFileLen);
  }

  // Set keywords_file 
  Module.setValue(ptr + offset, keywordsFileBuffer, 'i8*');
  offset += 4;

  // Set keywords_buf to 0 - we're using a file instead
  Module.setValue(ptr + offset, 0, 'i8*');
  offset += 4;

  // Set keywords_buf_size to 0
  Module.setValue(ptr + offset, 0, 'i32');
  offset += 4;

  return {
    ptr: ptr, 
    len: numBytes, 
    featConfig: featConfig, 
    modelConfig: modelConfig,
    keywordsFileBuffer: keywordsFileBuffer
  };
}

/**
 * Free configuration memory
 */
function freeConfig(config, Module) {
  if ('buffer' in config) {
    Module._free(config.buffer);
  }

  if ('transducer' in config) {
    freeConfig(config.transducer, Module);
  }

  if ('featConfig' in config) {
    freeConfig(config.featConfig, Module);
  }

  if ('modelConfig' in config) {
    freeConfig(config.modelConfig, Module);
  }

  if ('keywordsFileBuffer' in config && config.keywordsFileBuffer) {
    Module._free(config.keywordsFileBuffer);
  }

  Module._free(config.ptr);
}

/**
 * Global helper function to create a Kws instance
 */
function createKws(Module, myConfig) {
  let transducerConfig = {
    encoder: './encoder-epoch-12-avg-2-chunk-16-left-64.onnx',
    decoder: './decoder-epoch-12-avg-2-chunk-16-left-64.onnx',
    joiner: './joiner-epoch-12-avg-2-chunk-16-left-64.onnx',
  };
  
  let modelConfig = {
    transducer: transducerConfig,
    tokens: './tokens.txt',
    provider: 'cpu',
    modelType: '',
    numThreads: 1,
    debug: 1,
    modelingUnit: 'cjkchar',
    bpeVocab: '',
  };

  let featConfig = {
    samplingRate: 16000,
    featureDim: 80,
  };

  let configObj = {
    featConfig: featConfig,
    modelConfig: modelConfig,
    maxActivePaths: 4,
    numTrailingBlanks: 1,
    keywordsScore: 1.0,
    keywordsThreshold: 0.25,
    // Use keywordsFile instead of keywords
    keywordsFile: './keywords.txt'
  };

  if (myConfig) {
    configObj = myConfig;
  }
  return new Kws(configObj, Module);
} 