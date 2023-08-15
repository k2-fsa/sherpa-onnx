package com.k2fsa.sherpa.onnx

import android.content.res.AssetManager

fun main() {
    var featConfig = FeatureConfig(
        sampleRate = 16000,
        featureDim = 80,
    )

    // please refer to
    // https://k2-fsa.github.io/sherpa/onnx/pretrained_models/index.html
    // to dowload pre-trained models
    var modelConfig = OnlineModelConfig(
        transducer = OnlineTransducerModelConfig(
            encoder = "./sherpa-onnx-streaming-zipformer-en-2023-02-21/encoder-epoch-99-avg-1.onnx",
            decoder = "./sherpa-onnx-streaming-zipformer-en-2023-02-21/decoder-epoch-99-avg-1.onnx",
            joiner = "./sherpa-onnx-streaming-zipformer-en-2023-02-21/joiner-epoch-99-avg-1.onnx",
        ),
        tokens = "./sherpa-onnx-streaming-zipformer-en-2023-02-21/tokens.txt",
        numThreads = 1,
        debug = false,
    )

    var endpointConfig = EndpointConfig()

    var lmConfig = OnlineLMConfig()

    var config = OnlineRecognizerConfig(
        modelConfig = modelConfig,
        lmConfig = lmConfig,
        featConfig = featConfig,
        endpointConfig = endpointConfig,
        enableEndpoint = true,
        decodingMethod = "greedy_search",
        maxActivePaths = 4,
    )

    var model = SherpaOnnx(
        config = config,
    )

    var objArray = WaveReader.readWaveFromFile(
        filename = "./sherpa-onnx-streaming-zipformer-en-2023-02-21/test_wavs/0.wav",
    )
    var samples: FloatArray = objArray[0] as FloatArray
    var sampleRate: Int = objArray[1] as Int

    model.acceptWaveform(samples, sampleRate = sampleRate)
    while (model.isReady()) {
        model.decode()
    }

    var tailPaddings = FloatArray((sampleRate * 0.5).toInt()) // 0.5 seconds
    model.acceptWaveform(tailPaddings, sampleRate = sampleRate)
    model.inputFinished()
    while (model.isReady()) {
        model.decode()
    }

    println("results: ${model.text}")
}
