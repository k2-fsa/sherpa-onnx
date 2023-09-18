/*
 * // Copyright 2022-2023 by zhaoming
 */
/*
Real-time speech recognition from a microphone with com.k2fsa.sherpa.onnx Java API

example for cfgFile modelconfig.cfg
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
  enable_endpoint_detection=true
  decoding_method=greedy_search
  max_active_paths=4

*/
import com.k2fsa.sherpa.onnx.OnlineRecognizer;
import com.k2fsa.sherpa.onnx.OnlineStream;
import java.io.*;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.ShortBuffer;
import java.nio.charset.StandardCharsets;
import javax.sound.sampled.AudioFormat;
import javax.sound.sampled.AudioSystem;
import javax.sound.sampled.DataLine;
import javax.sound.sampled.TargetDataLine;

/** Microphone Example */
public class DecodeMic {
  MicRcgThread micRcgThread = null; // thread handle

  OnlineRecognizer rcgOjb; // the recognizer

  OnlineStream streamObj; // the stream

  public DecodeMic() {

    micRcgThread = new MicRcgThread(); // create a new instance for MicRcgThread
  }

  public void open() {
    micRcgThread.start(); // start to capture microphone data
  }

  public void close() {
    micRcgThread.stop(); // close capture
  }

  /** init asr engine with config file */
  public void initModelWithCfg(String cfgFile) {
    try {

      // set setSoPath() before running this
      rcgOjb = new OnlineRecognizer(cfgFile);

      streamObj = rcgOjb.createStream(); // create a stream for asr engine to feed data
    } catch (Exception e) {
      System.err.println(e);
      e.printStackTrace();
    }
  }

  /** read data from mic and feed to asr engine */
  class MicRcgThread implements Runnable {

    TargetDataLine capline; // line for capture mic data

    Thread thread; // this thread
    int segmentId = 0; // record the segment id when detect endpoint
    String preText = ""; // decoded text

    public MicRcgThread() {}

    public void start() {

      thread = new Thread(this);

      thread.start(); // start thread
    }

    public void stop() {
      capline.stop();
      capline.close();
      capline = null;
      thread = null;
    }

    /** feed captured microphone data to asr */
    public void decodeSample(byte[] samplebytes) {
      try {
        ByteBuffer byteBuf = ByteBuffer.wrap(samplebytes); // create a bytebuf for samples
        byteBuf.order(ByteOrder.LITTLE_ENDIAN); // set bytebuf to little endian
        ShortBuffer shortBuf = byteBuf.asShortBuffer(); // covert to short type
        short[] arrShort = new short[shortBuf.capacity()]; // array for copy short data
        float[] arrFloat = new float[shortBuf.capacity()]; // array for copy float data
        shortBuf.get(arrShort); // put date to arrShort

        for (int i = 0; i < arrShort.length; i++) {
          arrFloat[i] = arrShort[i] / 32768f; // loop to covert short data to float -1 to 1
        }
        streamObj.acceptWaveform(arrFloat); // feed asr engine with float data
        while (rcgOjb.isReady(streamObj)) { // if engine is ready for unprocessed data

          rcgOjb.decodeStream(streamObj); // decode for this stream
        }
        boolean isEndpoint =
            rcgOjb.isEndpoint(
                streamObj); // endpoint check, make sure enable_endpoint_detection=true in config
                            // file
        String nowText = rcgOjb.getResult(streamObj); // get asr result
        String recText = "";
        byte[] utf8Data; // for covert text to utf8
        if (isEndpoint && nowText.length() > 0) {
          rcgOjb.reSet(streamObj); // reSet stream when detect endpoint
          segmentId++;
          preText = nowText;
          recText = "text(seg_" + String.valueOf(segmentId) + "):" + nowText + "\n";
          utf8Data = recText.getBytes(StandardCharsets.UTF_8);
          System.out.println(new String(utf8Data));
        }

        if (!nowText.equals(preText)) { // if preText not equal nowtext
          preText = nowText;
          recText = nowText + "\n";
          utf8Data = recText.getBytes(StandardCharsets.UTF_8);
          System.out.println(new String(utf8Data));
        }
      } catch (Exception e) {
        System.err.println(e);
        e.printStackTrace();
      }
    }

    /** run mic capture thread */
    public void run() {
      System.out.println("Started! Please speak...");

      AudioFormat.Encoding encoding = AudioFormat.Encoding.PCM_SIGNED; // the pcm format
      float rate = 16000.0f; // using 16 kHz
      int channels = 1; // single channel
      int sampleSize = 16; // sampleSize 16bit
      boolean isBigEndian = false; // using little endian

      AudioFormat format =
          new AudioFormat(
              encoding, rate, sampleSize, channels, (sampleSize / 8) * channels, rate, isBigEndian);

      DataLine.Info info = new DataLine.Info(TargetDataLine.class, format);

      // check system support such data format
      if (!AudioSystem.isLineSupported(info)) {
        System.out.println(info + " not supported.");
        return;
      }

      // open a line for capture.

      try {
        capline = (TargetDataLine) AudioSystem.getLine(info);
        capline.open(format, capline.getBufferSize());
      } catch (Exception ex) {
        System.out.println(ex);
        return;
      }

      // the buf size for mic captured each time
      int bufferLengthInBytes = capline.getBufferSize() / 8 * format.getFrameSize();
      byte[] micData = new byte[bufferLengthInBytes];
      int numBytesRead;

      capline.start(); // start to capture mic data

      while (thread != null) {
        // read data from line
        if ((numBytesRead = capline.read(micData, 0, bufferLengthInBytes)) == -1) {
          break;
        }

        decodeSample(micData); // decode mic data
      }

      // stop and close

      try {
        if (capline != null) {
          capline.stop();
          capline.close();
          capline = null;
        }

      } catch (Exception ex) {
        System.err.println(ex);
      }
    }
  } // End class DecodeMic

  public static void main(String s[]) {
    try {
      String appDir = System.getProperty("user.dir");
      System.out.println("appdir=" + appDir);
      String cfgPath = appDir + "/modelconfig.cfg";
      String soPath = appDir + "/../build/lib/libsherpa-onnx-jni.so";
      OnlineRecognizer.setSoPath(soPath); // set so. lib for OnlineRecognizer

      DecodeMic decodeEx = new DecodeMic();
      decodeEx.initModelWithCfg(cfgPath); // init asr engine
      decodeEx.open(); // open thread for mic
      System.out.print("Press Enter to EXIT!\n");
      char i = (char) System.in.read();
      decodeEx.close();
    } catch (Exception e) {
      System.err.println(e);
      e.printStackTrace();
    }
  }
}
