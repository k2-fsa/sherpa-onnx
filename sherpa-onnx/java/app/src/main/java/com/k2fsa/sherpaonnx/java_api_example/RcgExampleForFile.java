/*
 * // Copyright 2022-2023 by zhaoming
 */
/*
Config modelconfig.cfg
  #set libsherpa-onnx-jni-java.so lib root dir
  solibpath=/sherpa-onnx/build/lib/libsherpa-onnx-jni-java.so   
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
package com.k2fsa.sherpaonnx.java_api_example;

import java.nio.charset.StandardCharsets;
import java.nio.charset.Charset;
import java.io.*;

import com.k2fsa.sherpaonnx.rcglib.OnlineRecognizer;
import com.k2fsa.sherpaonnx.rcglib.WavFile;

public class RcgExampleForFile {
    OnlineRecognizer rcgOjb;
    String wavfilename;

    public RcgExampleForFile(String filename) {

        wavfilename = filename;
    }

    public void initmodelwithpara() {
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

    }

    public void initmodelwithcfg() {
  
        //you should set OnlineRecognizer.setCfgPath(cfgpath) before running this
        rcgOjb = new OnlineRecognizer();

    }

    public void simpleExample() {
        //OnlineRecognizer rcgOjb=new OnlineRecognizer();
        try {
            WavFile wavFile = WavFile.openWavFile(new File(wavfilename));
            int numFrame = (int) wavFile.getNumFrames();
            float[] buffer = new float[numFrame];
            int framesRead = wavFile.readFrames(buffer, numFrame);
            //System.out.println("frameRead="+String.valueOf(framesRead));
            rcgOjb.acceptWaveform(buffer, 16000);
            rcgOjb.inputFinished();
            while (rcgOjb.isReady()) {
                rcgOjb.decode();
            }

            wavFile.close();
            String recText = "simple:" + rcgOjb.getText() + "\n";
            byte[] utf8Data = recText.getBytes(StandardCharsets.UTF_8);
            System.out.printf(new String(utf8Data));
            rcgOjb.reSet();
        } catch (Exception e) {
            System.err.println(e);
        }
    }

    public void streamExample() {
        try {
            WavFile wavFile = WavFile.openWavFile(new File(wavfilename));
            wavFile.display();
            // Get the number of audio channels in the wav file
            int numChannels = wavFile.getNumChannels();

            assert numChannels == 1; //only for single channel
            // Create a buffer of 16000 
            float[] buffer = new float[1600];
            int framesRead;
            float min = Float.MAX_VALUE;
            float max = Float.MIN_VALUE;
            int totalframes = 0;
            do {
                // Read frames into buffer
                framesRead = wavFile.readFrames(buffer, 1600);

                rcgOjb.acceptWaveform(buffer, 16000);
                if (rcgOjb.isReady()) {
                    rcgOjb.decode();
                }

                String testDate = rcgOjb.getText();
                byte[] utf8Data = testDate.getBytes(StandardCharsets.UTF_8);

                if (utf8Data.length > 0) {
                    System.out.printf(new String(utf8Data) + "\n");
                }
            }
            while (framesRead != 0);
            rcgOjb.inputFinished();
            while (rcgOjb.isReady()) {
                rcgOjb.decode();
            }
            // Close the wavFile
            wavFile.close();


            String wavText = "stream:" + rcgOjb.getText() + "\n";
            byte[] utf8Data = wavText.getBytes(StandardCharsets.UTF_8);


            System.out.printf(new String(utf8Data));
            rcgOjb.reSet();
        } catch (Exception e) {
            System.err.println(e);
        }
    }

    public static void main(String[] args) {
        try {
			String appdir=System.getProperty("user.dir");
			
			System.out.println("appdir="+appdir);
            String filename = appdir+"/test.wav";
			String cfgpath=appdir+"/modelconfig.cfg";
            OnlineRecognizer.setCfgPath(cfgpath);
            RcgExampleForFile rcgdemo = new RcgExampleForFile(filename);
            rcgdemo.initmodelwithcfg();
            rcgdemo.streamExample();
            //Thread.sleep(2000);
            rcgdemo.simpleExample();
        } catch (Exception e) {
            System.err.println(e);
			e.printStackTrace();
        }
        //rcgOjb.newRecognizer(cfg);

    }

}
