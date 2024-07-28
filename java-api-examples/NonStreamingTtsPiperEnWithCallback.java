// Copyright 2024 Xiaomi Corporation
//
// References
// https://www.baeldung.com/java-passing-method-parameter
// https://www.geeksforgeeks.org/how-to-create-a-thread-safe-queue-in-java/
// https://stackoverflow.com/questions/74077394/java-audio-how-to-continuously-write-bytes-to-an-audio-file-as-they-are-being-g

// This file shows how to use a piper VITS English TTS model
// to convert text to speech. You can pass a callback to the generation call,
// which is invoked whenever max_num_sentences sentences have been
// finished generation.
//
// The callback saves the generated samples into a queue, which are played
// by a separate thread.

import com.k2fsa.sherpa.onnx.*;
import java.util.Queue;
import java.util.concurrent.*;
import java.util.concurrent.ConcurrentLinkedQueue;
import javax.sound.sampled.*;

public class NonStreamingTtsPiperEn {
  public static void main(String[] args) {
    // please visit
    // https://github.com/k2-fsa/sherpa-onnx/releases/tag/tts-models
    // to download model files
    String model = "./vits-piper-en_GB-cori-medium/en_GB-cori-medium.onnx";
    String tokens = "./vits-piper-en_GB-cori-medium/tokens.txt";
    String dataDir = "./vits-piper-en_GB-cori-medium/espeak-ng-data";
    String text =
        "Today as always, men fall into two groups: slaves and free men. Whoever does not have"
            + " two-thirds of his day for himself, is a slave, whatever he may be: a statesman, a"
            + " businessman, an official, or a scholar.";

    OfflineTtsVitsModelConfig vitsModelConfig =
        OfflineTtsVitsModelConfig.builder()
            .setModel(model)
            .setTokens(tokens)
            .setDataDir(dataDir)
            .build();

    OfflineTtsModelConfig modelConfig =
        OfflineTtsModelConfig.builder()
            .setVits(vitsModelConfig)
            .setNumThreads(1)
            .setDebug(true)
            .build();

    OfflineTtsConfig config = OfflineTtsConfig.builder().setModel(modelConfig).build();
    OfflineTts tts = new OfflineTts(config);

    Queue<byte[]> samplesQueue = new ConcurrentLinkedQueue<>();

    Semaphore canPlaySem = new Semaphore(1);
    try {
      canPlaySem.acquire();
    } catch (InterruptedException ex) {
      System.out.println("Failed to acquire the play semaphore in the main thread");
      return;
    }

    Runnable playRuannable =
        () -> {
          try {
            canPlaySem.acquire();
          } catch (InterruptedException e) {
            System.out.println("Failed to get canPlay semaphore in the play thread");
            return;
          }

          // https://docs.oracle.com/javase/8/docs/api/javax/sound/sampled/AudioFormat.html
          AudioFormat format =
              new AudioFormat(
                  tts.getSampleRate(), // sampleRate
                  16, // sampleSizeInBits
                  1, // channels
                  true, // signed
                  false // bigEndian
                  );
          DataLine.Info info = new DataLine.Info(SourceDataLine.class, format);
          SourceDataLine line;
          try {
            line = (SourceDataLine) AudioSystem.getLine(info);

            int bufferSizeInBytes = tts.getSampleRate(); // 0.5 seconds
            line.open(format, bufferSizeInBytes);
          } catch (LineUnavailableException ex) {
            System.out.println("Failed to open a device for playing");
            return;
          }
          line.start();

          while (true) {
            if (samplesQueue.isEmpty()) {
              // Do nothing.
              //
              // If the generating speed is very slow, we can sleep
              // for some time here to save some CPU.
            } else {
              byte[] samples = samplesQueue.poll();
              if (samples.length == 1) {
                // end of the generating
                break;
              }
              line.write(samples, 0, samples.length);
            }
          }

          line.drain();
          line.close();
        };

    Thread playThread = new Thread(playRuannable);
    playThread.start();

    int sid = 0;
    float speed = 1.0f;
    long start = System.currentTimeMillis();
    GeneratedAudio audio =
        tts.generateWithCallback(
            text,
            sid,
            speed,
            (float[] samples) -> {

              // we use a byte array to save int16 samples
              byte[] samplesInt16 = new byte[samples.length * 2];
              for (int i = 0; i < samples.length; ++i) {
                float s = samples[i];
                if (s > 1) {
                  s = 1;
                }

                if (s < -1) {
                  s = -1;
                }

                short t = (short) (s * 32767);

                // we use little endian
                samplesInt16[2 * i] = (byte) (t & 0xff);
                samplesInt16[2 * i + 1] = (byte) ((t & 0xff00) >> 8);
              }

              samplesQueue.add(samplesInt16);

              canPlaySem.release();

              // Note: You can play the samples.
              // warning: You need to save a copy of samples since it is freed
              // when this function returns

              // return 1 to continue generation
              // return 0 to stop generation
              return 1;
            });

    // Since a sample always has two bytes. We put a single byte
    // into the queue to indicate that we have finished processing.
    samplesQueue.add(new byte[1]);

    long stop = System.currentTimeMillis();

    float timeElapsedSeconds = (stop - start) / 1000.0f;

    float audioDuration = audio.getSamples().length / (float) audio.getSampleRate();
    float real_time_factor = timeElapsedSeconds / audioDuration;

    try {
      playThread.join();
    } catch (InterruptedException ex) {
      System.out.println("Failed to join the play thread");
      return;
    }

    String waveFilename = "tts-piper-en.wav";
    audio.save(waveFilename);
    System.out.printf("-- elapsed : %.3f seconds\n", timeElapsedSeconds);
    System.out.printf("-- audio duration: %.3f seconds\n", timeElapsedSeconds);
    System.out.printf("-- real-time factor (RTF): %.3f\n", real_time_factor);
    System.out.printf("-- text: %s\n", text);
    System.out.printf("-- Saved to %s\n", waveFilename);

    tts.release();
  }
}
