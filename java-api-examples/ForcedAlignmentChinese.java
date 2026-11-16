// Copyright (c)  2026  Xiaomi Corporation

/*
 * Forced alignment of Chinese speech with the sherpa-onnx Java API.
 *
 * It aligns the test wav of the zh-aishell3-pinyin model package with its
 * transcript and writes a Praat TextGrid. The model directory, its
 * pronunciation lexicon and the test wav are hard-coded below.
 *
 * How to download the model (the package contains the model files, the
 * lexicon and the test wav used here):
 *
 *   wget https://github.com/k2-fsa/sherpa-onnx/releases/download/fa-models/zh-aishell3-pinyin.tar.bz2
 *   tar xf zh-aishell3-pinyin.tar.bz2
 *
 * How to run:
 *
 *   ./run-forced-alignment-chinese.sh
 *
 * Output:
 *
 *   zh.TextGrid in ForcedAlignmentChinese/. Open it with Praat
 *   (https://www.fon.hum.uva.nl/praat/) to visualize the phone and word
 *   boundaries.
 *
 * See also ForcedAlignmentEnglish.java and the Python examples
 * python-api-examples/forced-alignment-chinese.py.
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

public class ForcedAlignmentChinese {
    public static void main(String[] args) throws IOException {
        String modelDir = "./zh-aishell3-pinyin";
        String wav = modelDir + "/zh.wav";
        String text = "不要问你的国家能为你做什么，而要问你能为你的国家做什么。";

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

        Files.writeString(Path.of("zh.TextGrid"), result.toTextGrid());
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
