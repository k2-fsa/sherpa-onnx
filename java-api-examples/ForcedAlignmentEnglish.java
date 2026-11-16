// Copyright (c)  2026  Xiaomi Corporation

/*
 * Forced alignment of English speech with the sherpa-onnx Java API.
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
 *   ./run-forced-alignment-english.sh
 *
 * Output:
 *
 *   en.TextGrid in ForcedAlignmentEnglish/. Open it with Praat
 *   (https://www.fon.hum.uva.nl/praat/) to visualize the phone and word
 *   boundaries.
 *
 * See also ForcedAlignmentChinese.java and the Python examples
 * python-api-examples/forced-alignment-english.py.
 */

import com.k2fsa.sherpa.onnx.AlignmentInterval;
import com.k2fsa.sherpa.onnx.AlignmentTier;
import com.k2fsa.sherpa.onnx.ForcedAligner;
import com.k2fsa.sherpa.onnx.ForcedAlignerConfig;
import com.k2fsa.sherpa.onnx.ForcedAlignmentResult;
import com.k2fsa.sherpa.onnx.OfflineStream;
import com.k2fsa.sherpa.onnx.WaveReader;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

public class ForcedAlignmentEnglish {
    public static void main(String[] args) throws IOException {
        String modelDir = "./english-mfa-v3.1.0";
        String wav = modelDir + "/en.wav";
        String text =
                "ask not what your country can do for you, "
                        + "ask what you can do for your country.";

        ForcedAlignerConfig config = new ForcedAlignerConfig();
        config.modelDir = modelDir;
        config.lexicon = modelDir + "/lexicon.txt";

        long beginInit = System.nanoTime();
        ForcedAligner aligner = new ForcedAligner(config);
        OfflineStream stream = aligner.createStream();
        long endInit = System.nanoTime();

        WaveReader reader = new WaveReader(wav);
        // samples: float32 normalized to the range [-1, 1]
        stream.acceptWaveform(reader.getSamples(), reader.getSampleRate());

        long begin = System.nanoTime();
        ForcedAlignmentResult result = aligner.compute(stream, text);
        long end = System.nanoTime();

        for (AlignmentTier tier : result.getTiers()) {
            System.out.println("--- tier: " + tier.getName());
            for (AlignmentInterval interval : tier.getIntervals()) {
                System.out.printf("%.2f - %.2f  %s%n", interval.getStart(),
                        interval.getEnd(), interval.getText());
            }
        }

        Files.writeString(Path.of("en.TextGrid"), result.toTextGrid());
        System.out.println("wrote en.TextGrid");

        // time statistics, like sherpa-onnx-fa prints
        double initSeconds = (endInit - beginInit) / 1e9;
        double elapsedSeconds = (end - begin) / 1e9;
        double audioDuration =
                reader.getSamples().length / (double) reader.getSampleRate();
        System.out.printf("init is done in %.3f s%n", initSeconds);
        System.out.printf("Elapsed seconds: %.3f s%n", elapsedSeconds);
        System.out.printf(
                "Real time factor (RTF): %.3f / %.3f = %.3f%n",
                elapsedSeconds, audioDuration, elapsedSeconds / audioDuration);

        aligner.release();
    }
}
