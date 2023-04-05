/*
 * // Copyright 2022-2023 by zhaoming
 * // the online recognizer for sherpa-onnx, it can load config from a file
 * // or by argument
 */
/*
usage example:
        OnlineRecognizer.setCfgPath("/sherpa-onnx/sherpa-onnx/java/modelconfig.cfg"); //set cfg file
        OnlineRecognizer rcgOjb=new OnlineRecognizer();
		WavFile wavFile = WavFile.openWavFile(new File(wavfilename)); //read wav 
		int numFrame= (int) wavFile.getNumFrames(); //get wav size
		float[] buffer=new float[numFrame];
		int framesRead = wavFile.readFrames(buffer, numFrame);
		rcgOjb.acceptWaveform(buffer,16000);   //sample rate is 16000  //feed asr engine 
		rcgOjb.inputFinished();                //when all wav data is feed to engine
		while (rcgOjb.isReady()){rcgOjb.decode();}  //decode for text
		wavFile.close();
		String recText=rcgOjb.getText();      //get the text
        byte[] utf8Data = recText.getBytes(StandardCharsets.UTF_8);
        System.out.printf(new String(utf8Data));
*/
package com.k2fsa.sherpaonnx.rcglib;

import java.io.*;
import java.util.*;

public class OnlineRecognizer {

    private long ptr;  //this is the asr engine ptr 
    private String currentSid = "";  //each stream has a unique id, it is the id for current stream
    static private String cfgPath; //the config file, this file contains the model path and para , lib path  and so on.

    // load config file for OnlineRecognizer
    public OnlineRecognizer() {

        Map<String, String> proMap = OnlineRecognizer.readProperties();
        try {
            int sample_rate = Integer.parseInt(proMap.get("sample_rate").trim() );
            assert sample_rate == 16000; //only support for 16000
            EndpointRule rule1 = new EndpointRule(false, Float.parseFloat(proMap.get("rule1_min_trailing_silence").trim() ), 0.0F);
            EndpointRule rule2 = new EndpointRule(true, Float.parseFloat(proMap.get("rule2_min_trailing_silence").trim() ), 0.0F);
            EndpointRule rule3 = new EndpointRule(false, 0.0F, Float.parseFloat(proMap.get("rule3_min_utterance_length").trim() ));
            EndpointConfig end_cfg = new EndpointConfig(rule1, rule2, rule3);
            OnlineTransducerModelConfig model_cfg = new OnlineTransducerModelConfig(proMap.get("encoder").trim() ,
                    proMap.get("decoder").trim() , proMap.get("joiner").trim() , proMap.get("tokens").trim() , Integer.parseInt(proMap.get("num_threads").trim() ), false);
            FeatureConfig feat_config = new FeatureConfig(sample_rate, Integer.parseInt(proMap.get("feature_dim").trim() ));
            OnlineRecognizerConfig rcg_cfg = new OnlineRecognizerConfig(feat_config, model_cfg, end_cfg,
                    Boolean.parseBoolean(proMap.get("enable_endpoint_detection").trim() ),
                    proMap.get("decoding_method").trim() , Integer.parseInt(proMap.get("max_active_paths").trim() ));
            //create a new Recognizer 
            this.ptr = newRecognizer(rcg_cfg);
            currentSid = this.creatStream();  //set currentSid to the streamid

        } catch (Exception e) {
            System.err.println(e);
        }
    }

    // set  onlineRecognizer by parameter
    public OnlineRecognizer(String tokens, String encoder, String decoder, String joiner,
                            int num_threads,
                            int sample_rate,
                            int feature_dim,
                            boolean enable_endpoint_detection,
                            float rule1_min_trailing_silence,
                            float rule2_min_trailing_silence,
                            float rule3_min_utterance_length,
                            String decoding_method,
                            int max_active_paths) {

        assert sample_rate == 16000;  //only support for 16000
        EndpointRule rule1 = new EndpointRule(false, rule1_min_trailing_silence, 0.0F);
        EndpointRule rule2 = new EndpointRule(true, rule2_min_trailing_silence, 0.0F);
        EndpointRule rule3 = new EndpointRule(false, 0.0F, rule3_min_utterance_length);
        EndpointConfig end_cfg = new EndpointConfig(rule1, rule2, rule3);
        OnlineTransducerModelConfig model_cfg = new OnlineTransducerModelConfig(encoder, decoder, joiner, tokens, num_threads, false);
        FeatureConfig feat_config = new FeatureConfig(sample_rate, feature_dim);
        OnlineRecognizerConfig rcg_cfg = new OnlineRecognizerConfig(feat_config, model_cfg, end_cfg, enable_endpoint_detection, decoding_method, max_active_paths);
        this.ptr = newRecognizer(rcg_cfg);

    }

    private static Map<String, String> readProperties() {
        //read and parse config file
        Properties props = new Properties();
        Map<String, String> proMap = new HashMap<>();
        try {
            //System.out.println("cfgPath="+OnlineRecognizer.cfgPath);
            File file = new File(OnlineRecognizer.cfgPath);
            if (!file.exists()) {
                System.out.println("cfg file not exists!");
                System.exit(0);
            }
            InputStream in = new BufferedInputStream(new FileInputStream(OnlineRecognizer.cfgPath));
            props.load(in);
            Enumeration en = props.propertyNames();
            while (en.hasMoreElements()) {
                String key = (String) en.nextElement();
                String Property = props.getProperty(key);
                proMap.put(key, Property);
                //System.out.println(key+"="+Property);
            }

        } catch (Exception e) {
            e.printStackTrace();
        }
        return proMap;

    }

    public void acceptWaveform(float[] samples, int sampleRate) {
        //feed wave data to asr engine
        acceptWaveform(this.ptr, samples, sampleRate, currentSid);
    }

    public void decode() {
        //when feed samples to engine, call decode to let it process
        decode(this.ptr, currentSid);
    }

    public Boolean isEndpoint() {
        //check is it a endpoint?
        return isEndpoint(this.ptr, currentSid);
    }

    public Boolean isReady() {
        //check the engine whether is ready for decode
        return isReady(this.ptr, currentSid);
    }

    public void inputFinished() {
        //tell the engine all data are feeded
        inputFinished(this.ptr, currentSid);
    }

    public String getText() {
        //get text from the engine
        return getText(this.ptr, currentSid);
    }

    public void reSet() {
        //reset the stream, stream means wav data, one engine can process multiple streams in one time
        reset(this.ptr, currentSid);
        releaseStreams(currentSid);
        currentSid = creatStream(this.ptr);
    }

    public void decodeStreams() {
        //for decode parallel streams
        decodeStreams(this.ptr);
        System.out.println("not implemented!");
    }

    public String creatStream() {
        //create one stream for data to feed in
        String sid = creatStream(this.ptr);

        return sid;
    }

    public void releaseStreams(String sid) {
        //release one stream
        releaseStreams(this.ptr, sid);

    }

    public static void loadsolib() {
        //load the libsherpa-onnx-jni-java so. lib 
        Map<String, String> proMap = OnlineRecognizer.readProperties();
        //load .so lib from the path
		String so_path=proMap.get("solibpath").trim();
		System.out.println("lib path=" + so_path+"\n");
        System.load(so_path); //("sherpa-onnx-jni-java");


    }

    public static void setCfgPath(String cfgPath) {
        System.out.println("setCfgPath=" + cfgPath);
        OnlineRecognizer.cfgPath = cfgPath;
        OnlineRecognizer.loadsolib();
    }
    // JNI interface libsherpa-onnx-jni-java.so
    private native void acceptWaveform(long ptr, float[] samples, int sampleRate, String sid);

    private native void inputFinished(long ptr, String sid);

    private native String getText(long ptr, String sid);

    private native void reset(long ptr, String sid);

    private native void decode(long ptr, String sid);

    private native boolean isEndpoint(long ptr, String sid);

    private native boolean isReady(long ptr, String sid);

    private native long newRecognizer(OnlineRecognizerConfig config);

    private native void decodeStreams(long ptr);

    private native String creatStream(long ptr);

    private native void releaseStreams(long ptr, String sid);


}
