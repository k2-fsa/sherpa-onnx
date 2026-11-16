// Copyright (c)  2026  Xiaomi Corporation

/*
 * Forced alignment of English speech with the sherpa-onnx Kotlin API.
 *
 * It aligns the test wav of the english-mfa-v3.1.0 model package with its
 * transcript and writes a Praat TextGrid. The model directory, its
 * pronunciation lexicon and the test wav are hard-coded below.
 *
 * How to download the model (the package contains the model files, the
 * lexicon and the test wav used here):
 *
 *   wget https://github.com/k2-fsa/sherpa-onnx/releases/download/fa-models/english-mfa-v3.1.0.tar.bz2
 *   tar xf english-mfa-v3.1.0.tar.bz2
 *
 * How to run:
 *
 *   ./run.sh   (it runs testForcedAlignmentEnglish())
 *
 * Output:
 *
 *   en.TextGrid in test_forced_alignment_english/. Open it with Praat
 *   (https://www.fon.hum.uva.nl/praat/) to visualize the phone and word
 *   boundaries.
 *
 * See also ForcedAlignmentChinese.kt and the Python example
 * python-api-examples/forced-alignment-english.py.
 */

import com.k2fsa.sherpa.onnx.ForcedAligner
import com.k2fsa.sherpa.onnx.ForcedAlignerConfig
import com.k2fsa.sherpa.onnx.WaveReader
import java.io.File

fun main() {
    val modelDir = "./english-mfa-v3.1.0"
    val wav = "$modelDir/en.wav"
    val text = (
        "ask not what your country can do for you, " +
            "ask what you can do for your country."
        )

    val config = ForcedAlignerConfig(
        modelDir = modelDir,
        lexicon = "$modelDir/lexicon.txt",
    )

    val beginInit = System.nanoTime()
    val aligner = ForcedAligner(config = config)
    val stream = aligner.createStream()
    val endInit = System.nanoTime()

    val wave = WaveReader.readWave(wav)
    // wave.samples: float32 normalized to the range [-1, 1]
    stream.acceptWaveform(wave.samples, wave.sampleRate)

    val begin = System.nanoTime()
    val result = aligner.compute(stream, text)
    val end = System.nanoTime()

    for (tier in result.tiers) {
        println("--- tier: ${tier.name}")
        for (interval in tier.intervals) {
            println(
                "%.2f - %.2f  %s".format(
                    interval.start, interval.end, interval.text
                )
            )
        }
    }

    File("en.TextGrid").writeText(result.textGrid)
    println("wrote en.TextGrid")

    // time statistics, like sherpa-onnx-fa prints
    val initSeconds = (endInit - beginInit) / 1e9
    val elapsedSeconds = (end - begin) / 1e9
    val audioDuration = wave.samples.size.toDouble() / wave.sampleRate
    println("init is done in %.3f s".format(initSeconds))
    println("Elapsed seconds: %.3f s".format(elapsedSeconds))
    println(
        ("Real time factor (RTF): %.3f / %.3f = %.3f").format(
            elapsedSeconds, audioDuration, elapsedSeconds / audioDuration
        )
    )

    aligner.release()
}
