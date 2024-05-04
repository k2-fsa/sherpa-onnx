{
  'targets': [
    {
      'target_name': 'sherpa-onnx',
      'sources': [
        'src/sherpa-onnx-node-addon-api.cc',
        'src/streaming-asr.cc',
        'src/wave-reader.cc'
      ],
      'include_dirs': [
        "<!@(node -p \"require('node-addon-api').include\")",
        "<!@(pkg-config --variable=includedir sherpa-onnx)"
      ],
      'dependencies': ["<!(node -p \"require('node-addon-api').gyp\")"],
      'cflags!': [
        '-fno-exceptions',
      ],
      'cflags_cc!': [
        '-fno-exceptions',
        '-std=c++17'
      ],
      'libraries': [
        "<!@(pkg-config --libs sherpa-onnx)"
      ],
      'xcode_settings': {
        'GCC_ENABLE_CPP_EXCEPTIONS': 'YES',
        'CLANG_CXX_LIBRARY': 'libc++',
        'MACOSX_DEPLOYMENT_TARGET': '10.7'
      },
      'msvs_settings': {
        'VCCLCompilerTool': { 'ExceptionHandling': 1 },
      }
    }
  ]
}
