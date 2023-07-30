/*
 * // Copyright 2022-2023 by zhaoming
 */
/*
Config modelconfig.cfg
  sample_rate=16000
  feature_dim=80
  rule1_min_trailing_silence=2.4
  rule2_min_trailing_silence=1.2
  rule3_min_utterance_length=20
  encoder=/sherpa-onnx/build/bin/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/encoder-epoch-99-avg-1.onnx
  decoder=/sherpa-onnx/build/bin/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/decoder-epoch-99-avg-1.onnx
  joiner=/sherpa-onnx/build/bin/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/joiner-epoch-99-avg-1.onnx
  tokens=/sherpa-onnx/build/bin/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/tokens.txt
  num_threads=4
  enable_endpoint_detection=false
  decoding_method=greedy_search
  max_active_paths=4
*/

import com.k2fsa.sherpa.onnx.OnlineRecognizer;
import com.k2fsa.sherpa.onnx.OnlineStream;
import java.io.*;
import java.nio.charset.StandardCharsets;

public class DecodeFile {
  OnlineRecognizer rcgOjb;
  OnlineStream streamObj;
  String wavfilename;

  public DecodeFile(String fileName) {
    wavfilename = fileName;
  }

  public void initModelWithPara() {
    try {
      String modelDir =
          "/sherpa-onnx/build_old/bin/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20";
      String encoder = modelDir + "/encoder-epoch-99-avg-1.onnx";
      String decoder = modelDir + "/decoder-epoch-99-avg-1.onnx";
      String joiner = modelDir + "/joiner-epoch-99-avg-1.onnx";
      String tokens = modelDir + "/tokens.txt";
      int numThreads = 4;
      int sampleRate = 16000;
      int featureDim = 80;
      boolean enableEndpointDetection = false;
      float rule1MinTrailingSilence = 2.4F;
      float rule2MinTrailingSilence = 1.2F;
      float rule3MinUtteranceLength = 20F;
      String decodingMethod = "greedy_search";
      int maxActivePaths = 4;
      String lm_model = "";
      float lm_scale = 0.5F;
      String modelType = "zipformer";
      rcgOjb =
          new OnlineRecognizer(
              tokens,
              encoder,
              decoder,
              joiner,
              numThreads,
              sampleRate,
              featureDim,
              enableEndpointDetection,
              rule1MinTrailingSilence,
              rule2MinTrailingSilence,
              rule3MinUtteranceLength,
              decodingMethod,
              lm_model,
              lm_scale,
              maxActivePaths,
              modelType);
      streamObj = rcgOjb.createStream();
    } catch (Exception e) {
      System.err.println(e);
      e.printStackTrace();
    }
  }

  public void initModelWithCfg(String cfgFile) {
    try {
      // you should set setCfgPath() before running this
      rcgOjb = new OnlineRecognizer(cfgFile);
      streamObj = rcgOjb.createStream();
    } catch (Exception e) {
      System.err.println(e);
      e.printStackTrace();
    }
  }

  public void simpleExample() {
    try {
      float[] buffer = rcgOjb.readWavFile(wavfilename); // read data from file
      streamObj.acceptWaveform(buffer); // feed stream with data
      streamObj.inputFinished(); // tell engine you done with all data
      OnlineStream ssObj[] = new OnlineStream[1];
      while (rcgOjb.isReady(streamObj)) { // engine is ready for unprocessed data
        ssObj[0] = streamObj;
        rcgOjb.decodeStreams(ssObj); // decode for multiple stream
        // rcgOjb.DecodeStream(streamObj);   // decode for single stream
      }

      String recText = "simple:" + rcgOjb.getResult(streamObj) + "\n";
      byte[] utf8Data = recText.getBytes(StandardCharsets.UTF_8);
      System.out.println(new String(utf8Data));
      rcgOjb.reSet(streamObj);
      rcgOjb.releaseStream(streamObj); // release stream
      rcgOjb.release(); // release recognizer

    } catch (Exception e) {
      System.err.println(e);
      e.printStackTrace();
    }
  }

  public void streamExample() {
    try {
      float[] buffer = rcgOjb.readWavFile(wavfilename); // read data from file
      float[] chunk = new float[1600]; // //each time read 1600(0.1s) data
      int chunkIndex = 0;
      for (int i = 0; i < buffer.length; i++) // total wav length loop
      {
        chunk[chunkIndex] = buffer[i];
        chunkIndex++;
        if (chunkIndex >= 1600 || i == (buffer.length - 1)) {
          chunkIndex = 0;
          streamObj.acceptWaveform(chunk); // feed chunk
          if (rcgOjb.isReady(streamObj)) {
            rcgOjb.decodeStream(streamObj);
          }
          String testDate = rcgOjb.getResult(streamObj);
          byte[] utf8Data = testDate.getBytes(StandardCharsets.UTF_8);

          if (utf8Data.length > 0) {
            System.out.println(Float.valueOf((float) i / 16000) + ":" + new String(utf8Data));
          }
        }
      }
      streamObj.inputFinished();
      while (rcgOjb.isReady(streamObj)) {
        rcgOjb.decodeStream(streamObj);
      }

      String recText = "stream:" + rcgOjb.getResult(streamObj) + "\n";
      byte[] utf8Data = recText.getBytes(StandardCharsets.UTF_8);
      System.out.println(new String(utf8Data));
      rcgOjb.reSet(streamObj);
      rcgOjb.releaseStream(streamObj); // release stream
      rcgOjb.release(); // release recognizer

    } catch (Exception e) {
      System.err.println(e);
      e.printStackTrace();
    }
  }

  public static void main(String[] args) {
    try {
      String appDir = System.getProperty("user.dir");
      System.out.println("appdir=" + appDir);
      String fileName = appDir + "/test.wav";
      String cfgPath = appDir + "/modelconfig.cfg";
      String soPath = appDir + "/../build/lib/libsherpa-onnx-jni.so";
      OnlineRecognizer.setSoPath(soPath);
      DecodeFile rcgDemo = new DecodeFile(fileName);

      // ***************** */
      rcgDemo.initModelWithCfg(cfgPath);
      rcgDemo.streamExample();
      // **************** */
      rcgDemo.initModelWithCfg(cfgPath);
      rcgDemo.simpleExample();

    } catch (Exception e) {
      System.err.println(e);
      e.printStackTrace();
    }
  }
}
