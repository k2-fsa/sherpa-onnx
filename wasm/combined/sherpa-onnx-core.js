/**
 * sherpa-onnx-core.js
 * 
 * Core functionality for the SherpaOnnx WASM modules
 */

(function(global) {
  // Create main namespace
  const SherpaOnnx = {};
  
  // Check if Module already exists and extend it
  if (typeof global.Module !== 'undefined') {
    const originalOnRuntimeInitialized = global.Module.onRuntimeInitialized;
    global.Module.onRuntimeInitialized = function() {
      console.log("SherpaOnnx Core module initialized");
      if (originalOnRuntimeInitialized) originalOnRuntimeInitialized();
      if (global.onModuleReady) global.onModuleReady();
    };
  }
  
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
     * Safely create a directory in the WASM filesystem
     * Handles cases where the path already exists as a file
     * @param {string} dirPath - Path of the directory to create
     * @param {boolean} debug - Whether to output debug logs
     * @returns {boolean|Object} - True if successful or object with alternative path
     */
    safeCreateDirectory: function(dirPath, debug = false) {
      try {
        // Skip empty paths
        if (!dirPath || dirPath === '') {
          if (debug) console.log("Empty directory path, skipping");
          return true;
        }
        
        if (debug) console.log(`Creating directory: ${dirPath}`);
        
        // Generate a unique directory path to avoid conflicts
        const timestamp = Date.now();
        const random = Math.floor(Math.random() * 100000);
        const uniquePath = `temp_${timestamp}_${random}`;
        
        try {
          // Create the temporary directory first
          if (debug) console.log(`Creating unique temporary directory: ${uniquePath}`);
          global.Module.FS.mkdir(uniquePath, 0o777);
          
          // Then create our target directory inside the unique temp directory
          const safePath = `${uniquePath}/${dirPath}`;
          if (debug) console.log(`Creating directory in safe location: ${safePath}`);
          
          // Create all directories in the path
          let currentPath = uniquePath;
          const parts = dirPath.split('/');
          
          for (const part of parts) {
            if (!part) continue;
            currentPath += '/' + part;
            try {
              global.Module.FS.mkdir(currentPath, 0o777);
              if (debug) console.log(`Created directory component: ${currentPath}`);
            } catch (mkErr) {
              if (mkErr.errno !== 17) { // Not EEXIST
                console.error(`Error creating directory component ${currentPath}:`, mkErr);
                throw mkErr;
              }
            }
          }
          
          if (debug) console.log(`Successfully created nested directory at ${safePath}`);
          
          // Return the full alternative path
          return {
            success: true,
            altPath: safePath
          };
        } catch (nestErr) {
          console.error(`Failed to create nested directory structure:`, nestErr);
          
          // Try a different approach - directly creating a unique directory
          const directUniquePath = `${dirPath}_${timestamp}_${random}`;
          try {
            if (debug) console.log(`Trying direct unique path creation: ${directUniquePath}`);
            global.Module.FS.mkdir(directUniquePath, 0o777);
            if (debug) console.log(`Created unique directory: ${directUniquePath}`);
            
            return {
              success: true,
              altPath: directUniquePath
            };
          } catch (directErr) {
            console.error(`Failed to create directory with unique name:`, directErr);
            
            // Last attempt - try creating the original directory
            try {
              global.Module.FS.mkdir(dirPath, 0o777);
              if (debug) console.log(`Successfully created original directory: ${dirPath}`);
              return true;
            } catch (origErr) {
              // If it exists and is a directory, that's fine
              if (origErr.errno === 17) { // EEXIST
                try {
                  const stat = global.Module.FS.stat(dirPath);
                  if (stat.isDir) {
                    if (debug) console.log(`Directory ${dirPath} already exists`);
                    return true;
                  }
                } catch (statErr) {
                  console.error(`Error checking if ${dirPath} is a directory:`, statErr);
                }
              }
              
              console.error(`All attempts to create directory failed:`, origErr);
              throw origErr;
            }
          }
        }
      } catch (error) {
        console.error(`Failed to create directory ${dirPath}:`, error);
        return false;
      }
    },
    
    /**
     * Safely load a file from a URL into the WASM file system
     * Takes care of creating parent directories and verifying the file was written
     * @param {string} url - URL to fetch the file from
     * @param {string} localPath - Path where to save the file in WASM filesystem
     * @param {boolean} debug - Whether to output debug logs
     * @returns {Promise<boolean>} - True if successful, false otherwise
     */
    safeLoadFile: async function(url, localPath, debug = false) {
      try {
        if (debug) console.log(`Loading file from ${url} to ${localPath}`);
        
        // Get the directory
        const lastSlash = localPath.lastIndexOf('/');
        let targetPath = localPath;
        
        if (lastSlash > 0) {
          const dirPath = localPath.substring(0, lastSlash);
          if (debug) console.log(`Ensuring directory exists: ${dirPath}`);
          
          const dirResult = this.safeCreateDirectory(dirPath, debug);
          
          // Check if we need to use an alternate path
          if (dirResult && typeof dirResult === 'object' && dirResult.altPath) {
            // Adjust the target path to use the alternate directory path
            targetPath = `${dirResult.altPath}/${localPath.substring(lastSlash + 1)}`;
            if (debug) console.log(`Using alternate target path: ${targetPath}`);
          } else if (!dirResult) {
            throw new Error(`Failed to create directory ${dirPath}`);
          }
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
        try {
          global.Module.FS.writeFile(targetPath, new Uint8Array(buffer));
          
          // Verify the file was written
          try {
            const stat = global.Module.FS.stat(targetPath);
            if (debug) console.log(`File written to ${targetPath}, size: ${stat.size} bytes`);
          } catch (statErr) {
            throw new Error(`Failed to verify file was written: ${statErr.message}`);
          }
          
          // Return both the success status and the actual path used
          return {
            success: true,
            path: targetPath
          };
        } catch (writeErr) {
          console.error(`Error writing file to ${targetPath}:`, writeErr);
          throw writeErr;
        }
      } catch (error) {
        console.error(`Error loading ${url}:`, error);
        return false;
      }
    },
    
    /**
     * Check if a file exists in the WASM filesystem
     * @param {string} path - Path to check
     * @returns {boolean} - True if file exists, false otherwise
     */
    fileExists: function(path) {
      try {
        global.Module.FS.stat(path);
        return true;
      } catch (e) {
        return false;
      }
    },
    
    /**
     * Check if path exists and is a directory
     * @param {string} path - Path to check
     * @returns {boolean} - True if path exists and is a directory, false otherwise
     */
    isDirectory: function(path) {
      try {
        const stat = global.Module.FS.stat(path);
        return stat.isDir;
      } catch (e) {
        return false;
      }
    },
    
    /**
     * Remove a file or directory from the WASM filesystem
     * @param {string} path - Path to remove
     * @param {boolean} debug - Whether to output debug logs
     * @returns {boolean} - True if successful, false otherwise
     */
    removePath: function(path, debug = false) {
      try {
        if (!this.fileExists(path)) {
          if (debug) console.log(`Path ${path} doesn't exist, nothing to remove`);
          return true;
        }
        
        if (this.isDirectory(path)) {
          if (debug) console.log(`Removing directory ${path}`);
          global.Module.FS.rmdir(path);
        } else {
          if (debug) console.log(`Removing file ${path}`);
          global.Module.FS.unlink(path);
        }
        
        return true;
      } catch (error) {
        console.error(`Error removing path ${path}:`, error);
        return false;
      }
    },
    
    // Backward compatibility aliases - DIRECTLY COPY FUNCTIONALITY to avoid any reference issues
    ensureDirectory: function(dirPath) {
      console.log(`Using legacy ensureDirectory on path: ${dirPath}`);
      
      try {
        // Skip empty paths
        if (!dirPath || dirPath === '') {
          console.log("Empty directory path, skipping");
          return true;
        }
        
        console.log(`Creating directory: ${dirPath}`);
        
        // First check if the path exists
        try {
          const info = global.Module.FS.analyzePath(dirPath);
          if (info.exists) {
            const stat = global.Module.FS.stat(dirPath);
            if (stat.isDir) {
              console.log(`Directory ${dirPath} already exists`);
              return true;
            } else {
              // It exists as a file, remove it first
              console.log(`Path ${dirPath} exists as a file, removing it`);
              global.Module.FS.unlink(dirPath);
              // Then create as directory
              global.Module.FS.mkdir(dirPath, 0o777);
              console.log(`Successfully created directory at ${dirPath}`);
              return true;
            }
          } else {
            // Path doesn't exist, create it
            global.Module.FS.mkdir(dirPath, 0o777);
            console.log(`Created new directory at ${dirPath}`);
            return true;
          }
        } catch (e) {
          if (e.errno === 44 || e.errno === 2) { // ENOENT - path doesn't exist
            // Create the directory
            global.Module.FS.mkdir(dirPath, 0o777);
            console.log(`Created directory at ${dirPath}`);
            return true;
          } else if (e.errno === 17) { // EEXIST - already exists
            console.log(`Directory ${dirPath} already exists`);
            return true;
          } else {
            console.error(`Error creating directory ${dirPath}:`, e);
            throw e;
          }
        }
      } catch (error) {
        console.error(`Failed to create directory ${dirPath}:`, error);
        return false;
      }
    },
    
    loadFile: async function(url, localPath) {
      console.log(`DEBUG: DIRECT loadFile called with url: ${url}, localPath: ${localPath}`);
      
      try {
        console.log(`Loading file from ${url} to ${localPath}`);
        
        // Get the directory
        const lastSlash = localPath.lastIndexOf('/');
        if (lastSlash > 0) {
          const dirPath = localPath.substring(0, lastSlash);
          console.log(`Ensuring directory exists: ${dirPath}`);
          
          // Use the ensureDirectory function directly to avoid any reference issues
          this.ensureDirectory(dirPath);
        }
        
        // Fetch the file
        console.log(`Fetching ${url}`);
        const response = await fetch(url);
        
        if (!response.ok) {
          throw new Error(`Failed to fetch ${url}: ${response.status} ${response.statusText}`);
        }
        
        const buffer = await response.arrayBuffer();
        
        if (!buffer || buffer.byteLength === 0) {
          throw new Error(`Empty response from ${url}`);
        }
        
        console.log(`Downloaded ${url}, size: ${buffer.byteLength} bytes`);
        
        // Write the file
        try {
          global.Module.FS.writeFile(localPath, new Uint8Array(buffer));
          
          // Verify the file was written
          try {
            const stat = global.Module.FS.stat(localPath);
            console.log(`File written to ${localPath}, size: ${stat.size} bytes`);
          } catch (statErr) {
            throw new Error(`Failed to verify file was written: ${statErr.message}`);
          }
          
          return true;
        } catch (writeErr) {
          console.error(`Error writing file to ${localPath}:`, writeErr);
          throw writeErr;
        }
      } catch (error) {
        console.error(`Error loading ${url}:`, error);
        return false;
      }
    },
    
    /**
     * Create a model directory with a guaranteed unique path to avoid conflicts
     * Will create a completely new unique directory for models
     * 
     * @param {string} baseName - Base name for the model directory
     * @param {boolean} debug - Whether to enable debug logging
     * @returns {Promise<Object>} - Object with success status and path information
     */
    createModelDirectory: async function(baseName, debug = false) {
      try {
        if (!baseName || typeof baseName !== 'string') {
          baseName = 'model-dir';
        }
        
        // Generate a unique path with timestamp and random ID
        const timestamp = Date.now();
        const randomId = Math.floor(Math.random() * 1000000);
        const uniqueDirName = `${baseName}_${timestamp}_${randomId}`;
        
        if (debug) console.log(`Creating unique model directory: ${uniqueDirName}`);
        
        try {
          // Create the directory
          global.Module.FS.mkdir(uniqueDirName, 0o777);
          
          if (debug) console.log(`Successfully created unique model directory: ${uniqueDirName}`);
          
          return {
            success: true,
            baseName: baseName,
            uniquePath: uniqueDirName,
            timestamp: timestamp,
            randomId: randomId
          };
        } catch (error) {
          console.error(`Failed to create unique model directory: ${uniqueDirName}`, error);
          
          // Try a different random ID
          const newRandomId = Math.floor(Math.random() * 1000000);
          const backupDirName = `backup_${baseName}_${timestamp}_${newRandomId}`;
          
          if (debug) console.log(`Trying backup directory name: ${backupDirName}`);
          
          try {
            global.Module.FS.mkdir(backupDirName, 0o777);
            
            if (debug) console.log(`Successfully created backup model directory: ${backupDirName}`);
            
            return {
              success: true,
              baseName: baseName,
              uniquePath: backupDirName,
              timestamp: timestamp,
              randomId: newRandomId,
              isBackup: true
            };
          } catch (backupError) {
            console.error(`Failed to create backup model directory: ${backupDirName}`, backupError);
            return {
              success: false,
              error: backupError
            };
          }
        }
      } catch (error) {
        console.error(`Error in createModelDirectory:`, error);
        return {
          success: false,
          error: error
        };
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
        // Force clean the target path if it exists as a file
        try {
          const stat = global.Module.FS.stat(targetPath);
          const isFile = (stat.mode & 61440) === 32768;
          if (isFile) {
            if (debug) console.log(`Target path ${targetPath} exists as a FILE - removing it`);
            global.Module.FS.unlink(targetPath);
          }
        } catch (err) {
          // Path doesn't exist, which is fine
        }
        
        // Make sure the base directory exists
        try {
          this.mkdirp(targetPath);
          if (debug) console.log(`Created base directory ${targetPath}`);
        } catch (dirErr) {
          console.error(`Failed to create base directory ${targetPath}: ${dirErr.message}`);
          return { 
            success: false, 
            error: `Failed to create target directory: ${dirErr.message}` 
          };
        }
        
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
        const directories = new Set();
        for (const path in zip.files) {
          const file = zip.files[path];
          
          if (file.dir) {
            // Add directory path
            directories.add(`${targetPath}/${path}`);
          } else {
            // Add parent directory path for files
            const dirPath = path.substring(0, path.lastIndexOf('/'));
            if (dirPath) {
              directories.add(`${targetPath}/${dirPath}`);
            }
          }
        }
        
        // Create directories in sorted order to ensure parents are created first
        if (debug) console.log(`Creating ${directories.size} directories`);
        const sortedDirs = [...directories].sort((a, b) => a.split('/').length - b.split('/').length);
        for (const dir of sortedDirs) {
          try {
            this.mkdirp(dir);
          } catch (e) {
            console.warn(`Error creating directory ${dir}: ${e.message}`);
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
            FS.writeFile(fullPath, new Uint8Array(content));
            extractedFiles.push(fullPath);
            
            if (debug && extractedFiles.length % 50 === 0) {
              console.log(`Extracted ${extractedFiles.length} files so far...`);
            }
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
     * Create directory and parents if needed
     * @param {string} dirPath - Directory path
     */
    mkdirp: function(dirPath) {
      if (!dirPath || dirPath === '/') return;
      
      const parts = dirPath.split('/').filter(p => p);
      let current = '';
      
      for (const part of parts) {
        current += '/' + part;
        try {
          const stat = global.Module.FS.stat(current);
          // Only continue if it's a directory
          if ((stat.mode & 61440) !== 16384) { // Not a directory (S_IFDIR = 16384)
            console.error(`Path ${current} exists but is not a directory`);
            
            // Try to delete it if it's a file
            if ((stat.mode & 61440) === 32768) { // Is a file (S_IFREG = 32768)
              console.log(`Removing file at ${current} to create directory`);
              global.Module.FS.unlink(current);
              global.Module.FS.mkdir(current);
            } else {
              throw new Error(`Path exists but is not a directory: ${current}`);
            }
          }
        } catch (e) {
          // ENOENT error means directory doesn't exist, so create it
          if (e.errno === 44 || e.errno === 2 || e.message.includes('No such file or directory')) {
            try {
              global.Module.FS.mkdir(current);
            } catch (mkdirErr) {
              console.error(`Failed to create directory ${current}:`, mkdirErr);
              throw mkdirErr;
            }
          } else {
            console.error(`Error processing path ${current}:`, e);
            throw e; // Rethrow other errors
          }
        }
      }
      
      // Verify the directory was created
      try {
        const stat = global.Module.FS.stat(dirPath);
        if ((stat.mode & 61440) !== 16384) { // Not a directory
          throw new Error(`Path ${dirPath} was created but is not a directory`);
        }
      } catch (verifyErr) {
        console.error(`Failed to verify directory ${dirPath}:`, verifyErr);
        throw verifyErr;
      }
    },
    
    /**
     * Prepare a model directory and load files
     * @param {Array<Object>} files - List of files to prepare
     * @param {string} baseDir - Base directory for the model
     * @param {boolean} debug - Enable debug logging
     * @returns {Promise<Object>} - Result of the preparation
     */
    prepareModelDirectory: async function(files, baseDir = 'models', debug = false) {
      if (debug) console.log(`Preparing model directory with base: ${baseDir}`);
      
      try {
        // Create a unique directory name with random suffix to avoid conflicts
        const uniqueSuffix = Math.random().toString(36).substring(2, 10);
        const uniqueDir = `${baseDir}-${uniqueSuffix}`;
        
        if (debug) console.log(`Creating model directory: ${uniqueDir}`);
        
        // Force clean any problematic paths before creating new directories
        this.forceCleanPaths(baseDir, uniqueDir, debug);
        
        // Track results for each file
        const fileResults = [];
        
        // Process each file in the file list
        const archiveFiles = files.filter(f => f.isZip);
        const regularFiles = files.filter(f => !f.isZip);
        
        // First process all regular files to ensure the model directory is created
        for (const file of regularFiles) {
          try {
            if (file.content) {
              // Write string content directly to file
              const filename = this.joinPaths(uniqueDir, file.filename);
              const directoryPath = filename.substring(0, filename.lastIndexOf('/'));
              
              if (debug) console.log(`Writing content to ${filename}`);
              
              // Ensure the directory exists
              this.mkdirp(directoryPath);
              
              // Write the file
              FS.writeFile(filename, file.content);
              
              fileResults.push({
                success: true,
                path: filename,
                original: file
              });
            } else if (file.url) {
              // Load file from URL
              if (debug) console.log(`Fetching file from ${file.url}`);
              const response = await fetch(file.url);
              
              if (!response.ok) {
                console.error(`Failed to fetch ${file.url}: ${response.status} ${response.statusText}`);
                fileResults.push({
                  success: false,
                  error: `HTTP error: ${response.status}`,
                  original: file
                });
                continue;
              }
              
              // Write the downloaded file
              const filename = this.joinPaths(uniqueDir, file.filename);
              const directoryPath = filename.substring(0, filename.lastIndexOf('/'));
              
              if (debug) console.log(`Writing downloaded file to ${filename}`);
              
              // Ensure the directory exists
              this.mkdirp(directoryPath);
              
              // Get binary data and write to file
              const arrayBuffer = await response.arrayBuffer();
              FS.writeFile(filename, new Uint8Array(arrayBuffer), { encoding: 'binary' });
              
              fileResults.push({
                success: true,
                path: filename,
                original: file
              });
            } else {
              console.error('Invalid file specification: no content or URL');
              fileResults.push({
                success: false,
                error: 'Invalid file specification',
                original: file
              });
            }
          } catch (error) {
            console.error(`Error processing file: ${error.message}`);
            fileResults.push({
              success: false,
              error: error.message,
              original: file
            });
          }
        }
        
        // Now process archives with the correct model directory path
        for (const file of archiveFiles) {
          try {
            if (debug) console.log(`Fetching archive from ${file.url}`);
            const response = await fetch(file.url);
            
            if (!response.ok) {
              console.error(`Failed to fetch ${file.url}: ${response.status} ${response.statusText}`);
              fileResults.push({
                success: false,
                error: `HTTP error: ${response.status}`,
                original: file
              });
              continue;
            }
            
            if (debug) console.log(`Processing archive ${file.url}`);
            const zipData = await response.arrayBuffer();
            
            // Set the extract path to the created model directory if not specified
            const extractPath = file.extractToPath || uniqueDir;
            
            // Clean existing files if requested
            if (file.cleanBeforeExtract) {
              if (debug) console.log(`Cleaning before extraction at ${extractPath}`);
              try {
                // Create the directory if it doesn't exist
                this.mkdirp(extractPath);
                
                // Remove any existing espeak-ng-data directory
                const espeakDir = `${extractPath}/espeak-ng-data`;
                try {
                  FS.stat(espeakDir);
                  if (debug) console.log(`Removing existing directory: ${espeakDir}`);
                  this.removePath(espeakDir);
                } catch (e) {
                  // Directory doesn't exist, which is fine
                }
              } catch (cleanErr) {
                console.warn(`Could not clean extraction path: ${cleanErr.message}`);
              }
            }
            
            const extractResult = await this.extractZip(zipData, extractPath, debug);
            
            if (extractResult.success) {
              fileResults.push({
                success: true,
                path: extractPath,
                original: file,
                extractedFiles: extractResult.files
              });
            } else {
              fileResults.push({
                success: false,
                error: extractResult.error,
                original: file
              });
            }
          } catch (error) {
            console.error(`Error processing archive file: ${error.message}`);
            fileResults.push({
              success: false,
              error: error.message,
              original: file
            });
          }
        }
        
        // Check if any files failed to load
        const success = fileResults.some(result => result.success);
        
        if (debug) {
          console.log(`Model preparation ${success ? 'successful' : 'partially successful'}`);
          console.log(`Loaded ${fileResults.filter(r => r.success).length} of ${fileResults.length} files`);
        }
        
        return {
          modelDir: uniqueDir,
          success,
          files: fileResults
        };
      } catch (error) {
        console.error(`Error in prepareModelDirectory:`, error);
        return {
          success: false,
          error: error.message
        };
      }
    },
    
    /**
     * Join path segments properly
     * @param {...string} paths - Path segments to join
     * @returns {string} - Joined path
     */
    joinPaths: function(...paths) {
      return paths.join('/').replace(/\/+/g, '/');
    },
    
    /**
     * Ensure a directory exists, creating it if necessary
     * @param {string} dirPath - Directory path to ensure
     */
    ensureDirectory: function(dirPath) {
      if (!dirPath) return;
      
      // Skip if it's the root directory
      if (dirPath === '/') return;
      
      try {
        // Check if directory exists
        const stat = FS.stat(dirPath);
        if (stat.isDir) return; // Already exists - using isDir property, not a function
        throw new Error(`Path exists but is not a directory: ${dirPath}`);
      } catch (error) {
        // If error is that the path doesn't exist, create it
        if (error.errno === 44 || error.message.includes('No such file or directory')) {
          // Ensure parent directory exists first
          const parentDir = dirPath.split('/').slice(0, -1).join('/');
          if (parentDir) this.ensureDirectory(parentDir);
          
          // Create this directory
          FS.mkdir(dirPath);
          return;
        }
        
        // For other errors, rethrow
        throw error;
      }
    },
    
    /**
     * Debug the filesystem by listing key directories
     * @param {boolean} debug - Enable debug output
     */
    debugFilesystem: function(debug = true) {
      if (!debug) return;
      
      try {
        console.log("--- FILESYSTEM DEBUG ---");
        
        // List root directory
        const rootEntries = global.Module.FS.readdir('/');
        console.log("Root directory contents:", rootEntries);
        
        // Check relevant TTS model directories
        for (const entry of rootEntries) {
          // Only check TTS-related directories
          if (entry === 'tts-models' || entry.startsWith('tts-models-')) {
            try {
              const stat = global.Module.FS.stat('/' + entry);
              const isDir = (stat.mode & 61440) === 16384; // S_IFDIR
              
              if (isDir) {
                const subEntries = global.Module.FS.readdir('/' + entry);
                console.log(`Contents of /${entry}:`, subEntries);
              } else {
                console.log(`/${entry}: Not a directory`);
              }
            } catch (err) {
              console.log(`Error checking /${entry}:`, err.message);
            }
          }
        }
        
        console.log("--- END FILESYSTEM DEBUG ---");
      } catch (err) {
        console.log("Filesystem debug error:", err.message);
      }
    },
    
    /**
     * Clean up paths that may cause conflicts
     * @param {string} modelDir - The model directory 
     * @param {string} uniqueDir - The unique model directory
     * @param {boolean} debug - Enable debug output
     */
    forceCleanPaths: function(modelDir, uniqueDir, debug = false) {
      try {
        if (debug) console.log(`Cleaning paths: ${modelDir}, ${uniqueDir}`);
        
        // Clean up common problematic paths
        const pathsToClean = [
          '/espeak-ng-data',
          `/${modelDir}`,
          `/${uniqueDir}`
        ];
        
        for (const path of pathsToClean) {
          try {
            // Check if path exists
            const stat = global.Module.FS.stat(path);
            const type = stat.mode & 61440;
            const isFile = type === 32768;
            const isDir = type === 16384;
            
            if (isFile) {
              if (debug) console.log(`Removing file at ${path}`);
              global.Module.FS.unlink(path);
            } else if (isDir) {
              if (debug) console.log(`Removing directory at ${path}`);
              this.removePath(path, debug);
            }
          } catch (err) {
            // Path doesn't exist, which is fine
          }
        }
      } catch (err) {
        console.error("Error cleaning paths:", err.message);
      }
    }
  };
  
  // Expose SherpaOnnx to the global object
  global.SherpaOnnx = SherpaOnnx;
})(typeof window !== 'undefined' ? window : global); 