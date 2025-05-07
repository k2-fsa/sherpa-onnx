package com.k2fsa.sherpa.onnx

data class HomophoneReplacerConfig(
    var dictDir: String = "",
    var lexicon: String = "",
    var ruleFsts: String = "",
)
