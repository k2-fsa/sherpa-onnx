/**
 * sherpa-onnx-enhancement.js
 * 
 * Speech Enhancement functionality for SherpaOnnx
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
  
  // Create or use existing SpeechEnhancement namespace
  SherpaOnnx.SpeechEnhancement = SherpaOnnx.SpeechEnhancement || {};
  
  // Define the SpeechEnhancement module functionality
  SherpaOnnx.SpeechEnhancement = {
    /**
     * Load a Speech Enhancement model from URL
     * @param {Object} modelConfig - Configuration for the model
     * @returns {Promise<Object>} - Information about the loaded model
     */
    loadModel: async function(modelConfig) {
      const modelDir = modelConfig.modelDir || 'speech-enhancement-models';
      
      try {
        global.Module.FS.mkdir(modelDir, 0o777);
      } catch(e) {
        if (e.code !== 'EEXIST') throw e;
      }
      
      // Load the model
      await SherpaOnnx.FileSystem.loadFile(modelConfig.model || 'assets/enhancement/gtcrn.onnx', `${modelDir}/model.onnx`);
      
      return {
        modelDir: modelDir
      };
    },
    
    /**
     * Create a Speech Enhancement instance with the loaded model
     * @param {Object} loadedModel - Model information returned by loadModel
     * @param {Object} options - Additional configuration options
     * @returns {SpeechEnhancer} - A Speech Enhancement instance
     */
    createSpeechEnhancer: function(loadedModel, options = {}) {
      // This is a placeholder for actual implementation
      // In a real implementation, you would create the configuration
      // and pass it to the WASM module
      
      const config = {
        model: {
          gtcrn: {
            model: `${loadedModel.modelDir}/model.onnx`
          },
          numThreads: options.numThreads || 1,
          debug: options.debug !== undefined ? options.debug : 1,
          provider: options.provider || 'cpu'
        }
      };
      
      // In a real implementation, you would create and return an instance
      // of a SpeechEnhancer class
      
      console.warn('Speech Enhancement implementation is not fully functional yet');
      
      // Placeholder for the actual implementation
      return {
        config: config,
        
        // Placeholder methods that would normally interact with the WASM module
        process: function(audioSamples, sampleRate) {
          console.warn('SpeechEnhancement.process is a placeholder');
          return {
            enhancedSamples: audioSamples,  // Just return the original samples for now
            sampleRate: sampleRate
          };
        },
        
        free: function() {
          console.warn('SpeechEnhancement.free is a placeholder');
        }
      };
    }
  };
  
  // For Node.js environments
  if (typeof module !== 'undefined' && module.exports) {
    module.exports = SherpaOnnx;
  }
})(typeof window !== 'undefined' ? window : global); 