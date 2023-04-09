/*
 * // Copyright 2022-2023 by zhaoming
 */
/*
Config modelconfig.cfg
  #set libsherpa-onnx-jni.so lib root dir
  solibpath=/sherpa-onnx/build/lib/libsherpa-onnx-jni.so
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

import java.nio.charset.StandardCharsets;
import java.nio.charset.Charset;
import java.io.*;

import com.k2fsa.sherpaonnx.OnlineRecognizer;
import com.k2fsa.sherpaonnx.OnlineStream;

public class DecodeFile {
    OnlineRecognizer rcgOjb;
    OnlineStream streamObj;
    String wavfilename;

    public DecodeFile(String filename) {
        wavfilename = filename;
    }

    public void initModelWithPara() {
        try {
            String modelDir = "/sherpa-onnx/build_old/bin/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20";
            String encoder = modelDir + "/encoder-epoch-99-avg-1.onnx";
            String decoder = modelDir + "/decoder-epoch-99-avg-1.onnx";
            String joiner = modelDir + "/joiner-epoch-99-avg-1.onnx";
            String tokens = modelDir + "/tokens.txt";
            int num_threads = 4;
            int sample_rate = 16000;
            int feature_dim = 80;
            boolean enable_endpoint_detection = false;
            float rule1_min_trailing_silence = 2.4F;
            float rule2_min_trailing_silence = 1.2F;
            float rule3_min_utterance_length = 20F;
            String decoding_method = "greedy_search";
            int max_active_paths = 4;

            rcgOjb = new OnlineRecognizer(tokens, encoder, decoder, joiner, num_threads, sample_rate, feature_dim, enable_endpoint_detection, rule1_min_trailing_silence, rule2_min_trailing_silence, rule3_min_utterance_length, decoding_method, max_active_paths);
            streamObj = rcgOjb.CreateStream();
        } catch (Exception e) {
            System.err.println(e);
            e.printStackTrace();
        }
    }

    public void initModelWithCfg() {

        try {
            //you should set setCfgPath() before running this
            rcgOjb = new OnlineRecognizer();
            streamObj = rcgOjb.CreateStream();
        } catch (Exception e) {
            System.err.println(e);
            e.printStackTrace();
        }
    }

    public void simpleExample() {
        try {

            float[] buffer = rcgOjb.readWavFile(wavfilename); // read data from file
            streamObj.acceptWaveform(buffer, 16000);          //feed stream with data, and sample rate is 16000
            streamObj.inputFinished();                   //tell engine you done with all data
            while (rcgOjb.IsReady(streamObj)) {          //engine is ready for unprocessed data

                OnlineStream ssObj[] = new OnlineStream[1];
                ssObj[0] = streamObj;
                rcgOjb.DecodeStreams(ssObj);        //decode for multiple stream
                //rcgOjb.DecodeStream(streamObj);   //decode for single stream
            }

            String recText = "simple:" + rcgOjb.GetResult(streamObj) + "\n";
            byte[] utf8Data = recText.getBytes(StandardCharsets.UTF_8);
            System.out.println(new String(utf8Data));
            rcgOjb.Reset(streamObj);
            rcgOjb.releaseStream(streamObj);       //release stream
            rcgOjb.release();                      //release recognizer


        } catch (Exception e) {
            System.err.println(e);
            e.printStackTrace();
        }
    }

    public void streamExample() {

        try {

            float[] buffer = rcgOjb.readWavFile(wavfilename); // read data from file
            float[] chunk = new float[1600];       ////each time read 1600(0.1s) data
            int chunk_index = 0;
            for (int i = 0; i < buffer.length; i++)     //total wav length loop
            {
                chunk[chunk_index] = buffer[i];
                chunk_index++;
                if (chunk_index >= 1600 || i == (buffer.length - 1)) {
                    chunk_index = 0;
                    streamObj.acceptWaveform(chunk, 16000); //16000 is sample rate
                    if (rcgOjb.IsReady(streamObj)) {
                        rcgOjb.DecodeStream(streamObj);
                    }
                    String testDate = rcgOjb.GetResult(streamObj);
                    byte[] utf8Data = testDate.getBytes(StandardCharsets.UTF_8);

                    if (utf8Data.length > 0) {
                        System.out.println(Float.valueOf((float) i / 16000) + ":" + new String(utf8Data));
                    }
                }

            }
            streamObj.inputFinished();
            while (rcgOjb.IsReady(streamObj)) {
                rcgOjb.DecodeStream(streamObj);
            }


            String recText = "stream:" + rcgOjb.GetResult(streamObj) + "\n";
            byte[] utf8Data = recText.getBytes(StandardCharsets.UTF_8);
            System.out.println(new String(utf8Data));
            rcgOjb.Reset(streamObj);
            rcgOjb.releaseStream(streamObj);       //release stream
            rcgOjb.release();                      //release recognizer


        } catch (Exception e) {
            System.err.println(e);
            e.printStackTrace();
        }
    }

    public static void main(String[] args) {
        try {
            String appdir = System.getProperty("user.dir");
            System.out.println("appdir=" + appdir);
            String filename = appdir + "/test.wav";
            String cfgpath = appdir + "/modelconfig.cfg";
            OnlineRecognizer.setCfgPath(cfgpath);
            DecodeFile rcgdemo = new DecodeFile(filename);

            //***************** */
            rcgdemo.initModelWithCfg();
            rcgdemo.streamExample();
            //**************** */
            rcgdemo.initModelWithCfg();
            rcgdemo.simpleExample();

        } catch (Exception e) {
            System.err.println(e);
            e.printStackTrace();
        }


    }

}
