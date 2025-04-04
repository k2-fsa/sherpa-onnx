/**
 * sherpa-onnx-speaker.js
 * 
 * Speaker Diarization functionality for SherpaOnnx
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
  
  // Create or use existing SpeakerDiarization namespace
  SherpaOnnx.SpeakerDiarization = SherpaOnnx.SpeakerDiarization || {};
  
  // Define the SpeakerDiarization module functionality
  SherpaOnnx.SpeakerDiarization = {
    /**
     * Load Speaker Diarization models from URLs
     * @param {Object} modelConfig - Configuration for the models
     * @returns {Promise<Object>} - Information about the loaded models
     */
    loadModel: async function(modelConfig) {
      const modelDir = modelConfig.modelDir || 'speaker-diarization-models';
      
      try {
        global.Module.FS.mkdir(modelDir, 0o777);
      } catch(e) {
        if (e.code !== 'EEXIST') throw e;
      }
      
      // Load segmentation and embedding models
      await Promise.all([
        SherpaOnnx.FileSystem.loadFile(modelConfig.segmentation || 'assets/speakers/segmentation.onnx', `${modelDir}/segmentation.onnx`),
        SherpaOnnx.FileSystem.loadFile(modelConfig.embedding || 'assets/speakers/embedding.onnx', `${modelDir}/embedding.onnx`)
      ]);
      
      return {
        modelDir: modelDir
      };
    },
    
    /**
     * Create a Speaker Diarization instance with the loaded models
     * @param {Object} loadedModel - Model information returned by loadModel
     * @param {Object} options - Additional configuration options
     * @returns {SpeakerDiarization} - A Speaker Diarization instance
     */
    createSpeakerDiarization: function(loadedModel, options = {}) {
      // This is a placeholder for actual implementation
      // In a real implementation, you would create the configuration
      // and pass it to the WASM module
      
      const config = {
        segmentation: {
          pyannote: {
            model: `${loadedModel.modelDir}/segmentation.onnx`
          },
          numThreads: options.numThreads || 1,
          debug: options.debug !== undefined ? options.debug : 1,
          provider: options.provider || 'cpu'
        },
        embedding: {
          model: `${loadedModel.modelDir}/embedding.onnx`,
          numThreads: options.numThreads || 1,
          debug: options.debug !== undefined ? options.debug : 1,
          provider: options.provider || 'cpu'
        },
        clustering: {
          numClusters: options.numClusters || 0,  // 0 means auto-detect
          threshold: options.threshold || 0.8
        },
        minDurationOn: options.minDurationOn || 0.5,
        minDurationOff: options.minDurationOff || 0.5
      };
      
      // In a real implementation, you would create and return an instance
      // of a SpeakerDiarization class
      
      console.warn('Speaker Diarization implementation is not fully functional yet');
      
      // Placeholder for the actual implementation
      return {
        config: config,
        
        // Placeholder methods that would normally interact with the WASM module
        process: function(audioSamples, sampleRate) {
          console.warn('SpeakerDiarization.process is a placeholder');
          return {
            segments: []
          };
        },
        
        free: function() {
          console.warn('SpeakerDiarization.free is a placeholder');
        }
      };
    }
  };
  
  // For Node.js environments
  if (typeof module !== 'undefined' && module.exports) {
    module.exports = SherpaOnnx;
  }
})(typeof window !== 'undefined' ? window : global); 