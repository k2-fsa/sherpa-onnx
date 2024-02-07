

Module = {};
Module.onRuntimeInitialized = function() {
  console.log('inited!');
  console.log('module', Module);
  initSherpaOnnxOfflineTts()
};
