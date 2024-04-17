package com.k2fsa.sherpa.onnx

import android.content.res.AssetManager

fun callback(samples: FloatArray): Unit {
  println("callback got called with ${samples.size} samples");
}

fun main() {
  testSpokenLanguageIdentifcation()
  testAudioTagging()
  testSpeakerRecognition()
  testTts()
  testAsr("transducer")
  testAsr("zipformer2-ctc")
}

fun testSpokenLanguageIdentifcation() {
  val config = SpokenLanguageIdentificationConfig(
    whisper = SpokenLanguageIdentificationWhisperConfig(
      encoder = "./sherpa-onnx-whisper-tiny/tiny-encoder.int8.onnx",
      decoder = "./sherpa-onnx-whisper-tiny/tiny-decoder.int8.onnx",
      tailPaddings = 33,
    ),
    numThreads=1,
    debug=true,
    provider="cpu",
  )
  val slid = SpokenLanguageIdentification(assetManager=null, config=config)

  val testFiles = arrayOf(
    "./spoken-language-identification-test-wavs/ar-arabic.wav",
    "./spoken-language-identification-test-wavs/bg-bulgarian.wav",
    "./spoken-language-identification-test-wavs/de-german.wav",
  )

  for (waveFilename in testFiles) {
    val objArray = WaveReader.readWaveFromFile(
        filename = waveFilename,
    )
    val samples: FloatArray = objArray[0] as FloatArray
    val sampleRate: Int = objArray[1] as Int

    val stream = slid.createStream()
    stream.acceptWaveform(samples, sampleRate = sampleRate)
    val lang = slid.compute(stream)
    stream.release()
    println(waveFilename)
    println(lang)
  }
}

fun testAudioTagging() {
  val config = AudioTaggingConfig(
      model=AudioTaggingModelConfig(
        zipformer=OfflineZipformerAudioTaggingModelConfig(
          model="./sherpa-onnx-zipformer-audio-tagging-2024-04-09/model.int8.onnx",
        ),
        numThreads=1,
        debug=true,
        provider="cpu",
      ),
      labels="./sherpa-onnx-zipformer-audio-tagging-2024-04-09/class_labels_indices.csv",
      topK=5,
   )
  val tagger = AudioTagging(assetManager=null, config=config)

  val testFiles = arrayOf(
    "./sherpa-onnx-zipformer-audio-tagging-2024-04-09/test_wavs/1.wav",
    "./sherpa-onnx-zipformer-audio-tagging-2024-04-09/test_wavs/2.wav",
    "./sherpa-onnx-zipformer-audio-tagging-2024-04-09/test_wavs/3.wav",
    "./sherpa-onnx-zipformer-audio-tagging-2024-04-09/test_wavs/4.wav",
  )
  println("----------")
  for (waveFilename in testFiles) {
    val stream = tagger.createStream()

    val objArray = WaveReader.readWaveFromFile(
        filename = waveFilename,
    )
    val samples: FloatArray = objArray[0] as FloatArray
    val sampleRate: Int = objArray[1] as Int

    stream.acceptWaveform(samples, sampleRate = sampleRate)
    val events = tagger.compute(stream)
    stream.release()

    println(waveFilename)
    println(events)
    println("----------")
  }

  tagger.release()
}

fun computeEmbedding(extractor: SpeakerEmbeddingExtractor, filename: String): FloatArray {
    var objArray = WaveReader.readWaveFromFile(
        filename = filename,
    )
    var samples: FloatArray = objArray[0] as FloatArray
    var sampleRate: Int = objArray[1] as Int

    val stream = extractor.createStream()
    stream.acceptWaveform(sampleRate = sampleRate, samples=samples)
    stream.inputFinished()
    check(extractor.isReady(stream))

    val embedding = extractor.compute(stream)

    stream.release()

    return embedding
}

fun testSpeakerRecognition() {
    val config = SpeakerEmbeddingExtractorConfig(
        model="./3dspeaker_speech_eres2net_large_sv_zh-cn_3dspeaker_16k.onnx",
        )
    val extractor = SpeakerEmbeddingExtractor(config = config)

    val embedding1a = computeEmbedding(extractor, "./speaker1_a_cn_16k.wav")
    val embedding2a = computeEmbedding(extractor, "./speaker2_a_cn_16k.wav")
    val embedding1b = computeEmbedding(extractor, "./speaker1_b_cn_16k.wav")

    var manager = SpeakerEmbeddingManager(extractor.dim())
    var ok = manager.add(name = "speaker1", embedding=embedding1a)
    check(ok)

    manager.add(name = "speaker2", embedding=embedding2a)
    check(ok)

    var name = manager.search(embedding=embedding1b, threshold=0.5f)
    check(name == "speaker1")

    manager.release()

    manager = SpeakerEmbeddingManager(extractor.dim())
    val embeddingList = mutableListOf(embedding1a, embedding1b)
    ok = manager.add(name = "s1", embedding=embeddingList.toTypedArray())
    check(ok)

    name = manager.search(embedding=embedding1b, threshold=0.5f)
    check(name == "s1")

    name = manager.search(embedding=embedding2a, threshold=0.5f)
    check(name.length == 0)

    manager.release()
}

fun testTts() {
  // see https://github.com/k2-fsa/sherpa-onnx/releases/tag/tts-models
  // https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-en_US-amy-low.tar.bz2
  var config = OfflineTtsConfig(
    model=OfflineTtsModelConfig(
      vits=OfflineTtsVitsModelConfig(
        model="./vits-piper-en_US-amy-low/en_US-amy-low.onnx",
        tokens="./vits-piper-en_US-amy-low/tokens.txt",
        dataDir="./vits-piper-en_US-amy-low/espeak-ng-data",
      ),
      numThreads=1,
      debug=true,
    )
  )
  val tts = OfflineTts(config=config)
  val audio = tts.generateWithCallback(text="“Today as always, men fall into two groups: slaves and free men. Whoever does not have two-thirds of his day for himself, is a slave, whatever he may be: a statesman, a businessman, an official, or a scholar.”", callback=::callback)
  audio.save(filename="test-en.wav")
}

fun testAsr(type: String) {
    var featConfig = FeatureConfig(
        sampleRate = 16000,
        featureDim = 80,
    )

    var waveFilename: String
    var modelConfig: OnlineModelConfig = when (type) {
      "transducer" -> {
        waveFilename = "./sherpa-onnx-streaming-zipformer-en-2023-02-21/test_wavs/0.wav"
        // please refer to
        // https://k2-fsa.github.io/sherpa/onnx/pretrained_models/index.html
        // to dowload pre-trained models
        OnlineModelConfig(
            transducer = OnlineTransducerModelConfig(
                encoder = "./sherpa-onnx-streaming-zipformer-en-2023-02-21/encoder-epoch-99-avg-1.onnx",
                decoder = "./sherpa-onnx-streaming-zipformer-en-2023-02-21/decoder-epoch-99-avg-1.onnx",
                joiner = "./sherpa-onnx-streaming-zipformer-en-2023-02-21/joiner-epoch-99-avg-1.onnx",
            ),
            tokens = "./sherpa-onnx-streaming-zipformer-en-2023-02-21/tokens.txt",
            numThreads = 1,
            debug = false,
        )
      }
      "zipformer2-ctc" -> {
        waveFilename = "./sherpa-onnx-streaming-zipformer-ctc-multi-zh-hans-2023-12-13/test_wavs/DEV_T0000000000.wav"
        OnlineModelConfig(
            zipformer2Ctc = OnlineZipformer2CtcModelConfig(
                model = "./sherpa-onnx-streaming-zipformer-ctc-multi-zh-hans-2023-12-13/ctc-epoch-20-avg-1-chunk-16-left-128.onnx",
            ),
            tokens = "./sherpa-onnx-streaming-zipformer-ctc-multi-zh-hans-2023-12-13/tokens.txt",
            numThreads = 1,
            debug = false,
        )
      }
      else -> throw IllegalArgumentException(type)
    }

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
        filename = waveFilename,
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
