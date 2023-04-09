/*
 * // Copyright 2022-2023 by zhaoming
 * // the online recognizer for sherpa-onnx, it can load config from a file
 * // or by argument
 */
/*
usage example:
    String cfgpath=appdir+"/modelconfig.cfg";
    OnlineRecognizer.setCfgPath(cfgpath);      //set model config file
    OnlineRecognizer rcgOjb = new OnlineRecognizer();   //create a recognizer
    CreateStream streamObj=rcgOjb.CreateStream();       //create a stream for read wav data
    float[] buffer = rcgOjb.readWavFile(wavfilename); // read data from file
    streamObj.acceptWaveform(buffer, 16000);          //feed stream with data, and sample rate is 16000
    streamObj.inputFinished();                   //tell engine you done with all data 
    while (rcgOjb.IsReady(streamObj)) {          //engine is ready for unprocessed data

                OnlineStream ssObj[]=new OnlineStream[1];
                ssObj[0]=streamObj;
                rcgOjb.DecodeStreams(ssObj);        //decode for multiple stream decode
                //rcgOjb.DecodeStream(streamObj);   //decode for single stream decode
            }

    String recText = "simple:" + rcgOjb.GetResult(streamObj) + "\n";
    byte[] utf8Data = recText.getBytes(StandardCharsets.UTF_8);
    System.out.println(new String(utf8Data));
    rcgOjb.Reset(streamObj);
    rcgOjb.releaseStream(streamObj);       //release stream
    rcgOjb.release();                      //release recognizer

*/
package com.k2fsa.sherpaonnx;

import java.io.*;
import java.util.*;

public class OnlineRecognizer {

    private long ptr = 0; // this is the asr engine ptrss

    static private String cfgPath; // the config file, this file contains the model path and para , lib path and so
    // on.

    // load config file for OnlineRecognizer
    public OnlineRecognizer() {

        Map<String, String> proMap = OnlineRecognizer.readProperties();
        try {
            int sample_rate = Integer.parseInt(proMap.get("sample_rate").trim());
            assert sample_rate == 16000; // only support for 16000
            EndpointRule rule1 = new EndpointRule(false,
                    Float.parseFloat(proMap.get("rule1_min_trailing_silence").trim()), 0.0F);
            EndpointRule rule2 = new EndpointRule(true,
                    Float.parseFloat(proMap.get("rule2_min_trailing_silence").trim()), 0.0F);
            EndpointRule rule3 = new EndpointRule(false, 0.0F,
                    Float.parseFloat(proMap.get("rule3_min_utterance_length").trim()));
            EndpointConfig end_cfg = new EndpointConfig(rule1, rule2, rule3);
            OnlineTransducerModelConfig model_cfg = new OnlineTransducerModelConfig(proMap.get("encoder").trim(),
                    proMap.get("decoder").trim(), proMap.get("joiner").trim(), proMap.get("tokens").trim(),
                    Integer.parseInt(proMap.get("num_threads").trim()), false);
            FeatureConfig feat_config = new FeatureConfig(sample_rate,
                    Integer.parseInt(proMap.get("feature_dim").trim()));
            OnlineRecognizerConfig rcg_cfg = new OnlineRecognizerConfig(feat_config, model_cfg, end_cfg,
                    Boolean.parseBoolean(proMap.get("enable_endpoint_detection").trim()),
                    proMap.get("decoding_method").trim(), Integer.parseInt(proMap.get("max_active_paths").trim()));
            // create a new Recognizer, first parameter kept for android asset_manager ANDROID_API__ >= 9
            this.ptr = CreateOnlineRecognizer(new Object(), rcg_cfg);

        } catch (Exception e) {
            System.err.println(e);
        }
    }

    // use for android asset_manager ANDROID_API__ >= 9
    public OnlineRecognizer(Object asset_manager) {

        Map<String, String> proMap = OnlineRecognizer.readProperties();
        try {
            int sample_rate = Integer.parseInt(proMap.get("sample_rate").trim());
            assert sample_rate == 16000; // only support for 16000
            EndpointRule rule1 = new EndpointRule(false,
                    Float.parseFloat(proMap.get("rule1_min_trailing_silence").trim()), 0.0F);
            EndpointRule rule2 = new EndpointRule(true,
                    Float.parseFloat(proMap.get("rule2_min_trailing_silence").trim()), 0.0F);
            EndpointRule rule3 = new EndpointRule(false, 0.0F,
                    Float.parseFloat(proMap.get("rule3_min_utterance_length").trim()));
            EndpointConfig end_cfg = new EndpointConfig(rule1, rule2, rule3);
            OnlineTransducerModelConfig model_cfg = new OnlineTransducerModelConfig(proMap.get("encoder").trim(),
                    proMap.get("decoder").trim(), proMap.get("joiner").trim(), proMap.get("tokens").trim(),
                    Integer.parseInt(proMap.get("num_threads").trim()), false);
            FeatureConfig feat_config = new FeatureConfig(sample_rate,
                    Integer.parseInt(proMap.get("feature_dim").trim()));
            OnlineRecognizerConfig rcg_cfg = new OnlineRecognizerConfig(feat_config, model_cfg, end_cfg,
                    Boolean.parseBoolean(proMap.get("enable_endpoint_detection").trim()),
                    proMap.get("decoding_method").trim(), Integer.parseInt(proMap.get("max_active_paths").trim()));
            // create a new Recognizer, first parameter kept for android asset_manager ANDROID_API__ >= 9
            this.ptr = CreateOnlineRecognizer(asset_manager, rcg_cfg);

        } catch (Exception e) {
            System.err.println(e);
        }
    }


    // set onlineRecognizer by parameter
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

        assert sample_rate == 16000; // only support for 16000 now
        EndpointRule rule1 = new EndpointRule(false, rule1_min_trailing_silence, 0.0F);
        EndpointRule rule2 = new EndpointRule(true, rule2_min_trailing_silence, 0.0F);
        EndpointRule rule3 = new EndpointRule(false, 0.0F, rule3_min_utterance_length);
        EndpointConfig end_cfg = new EndpointConfig(rule1, rule2, rule3);
        OnlineTransducerModelConfig model_cfg = new OnlineTransducerModelConfig(encoder, decoder, joiner, tokens,
                num_threads, false);
        FeatureConfig feat_config = new FeatureConfig(sample_rate, feature_dim);
        OnlineRecognizerConfig rcg_cfg = new OnlineRecognizerConfig(feat_config, model_cfg, end_cfg,
                enable_endpoint_detection, decoding_method, max_active_paths);
        // create a new Recognizer, first parameter kept for android asset_manager ANDROID_API__ >= 9
        this.ptr = CreateOnlineRecognizer(new Object(), rcg_cfg);

    }

    public static Map<String, String> readProperties() {
        // read and parse config file
        Properties props = new Properties();
        Map<String, String> proMap = new HashMap<>();
        try {

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
                // System.out.println(key+"="+Property);
            }

        } catch (Exception e) {
            e.printStackTrace();
        }
        return proMap;

    }

    public void DecodeStream(OnlineStream s) throws Exception {
        if (this.ptr == 0) throw new Exception("null exception for recognizer ptr");
        long s_ptr = s.getS_ptr();
        if (s_ptr == 0) throw new Exception("null exception for stream s_ptr");
        // when feeded samples to engine, call DecodeStream to let it process
        DecodeStream(this.ptr, s_ptr);
    }

    public void DecodeStreams(OnlineStream[] ssOjb) throws Exception {
        if (this.ptr == 0) throw new Exception("null exception for recognizer ptr");
        // decode for multiple streams
        long[] ss = new long[ssOjb.length];
        for (int i = 0; i < ssOjb.length; i++) {
            ss[i] = ssOjb[i].getS_ptr();
            if (ss[i] == 0) throw new Exception("null exception for stream s_ptr");
        }
        DecodeStreams(this.ptr, ss);
    }

    public boolean IsReady(OnlineStream s) throws Exception {
        // whether the engine is ready for decode
        if (this.ptr == 0) throw new Exception("null exception for recognizer ptr");
        long s_ptr = s.getS_ptr();
        if (s_ptr == 0) throw new Exception("null exception for stream s_ptr");
        return IsReady(this.ptr, s_ptr);
    }

    public String GetResult(OnlineStream s) throws Exception {
        // get text from the engine
        if (this.ptr == 0) throw new Exception("null exception for recognizer ptr");
        long s_ptr = s.getS_ptr();
        if (s_ptr == 0) throw new Exception("null exception for stream s_ptr");
        return GetResult(this.ptr, s_ptr);
    }

    public boolean IsEndpoint(OnlineStream s) throws Exception {
        if (this.ptr == 0) throw new Exception("null exception for recognizer ptr");
        long s_ptr = s.getS_ptr();
        if (s_ptr == 0) throw new Exception("null exception for stream s_ptr");
        return IsEndpoint(this.ptr, s_ptr);
    }

    public void Reset(OnlineStream s) throws Exception {
        if (this.ptr == 0) throw new Exception("null exception for recognizer ptr");
        long s_ptr = s.getS_ptr();
        if (s_ptr == 0) throw new Exception("null exception for stream s_ptr");
        Reset(this.ptr, s_ptr);
    }

    public OnlineStream CreateStream() throws Exception {
        // create one stream for data to feed in
        if (this.ptr == 0) throw new Exception("null exception for recognizer ptr");
        long s_ptr = CreateStream(this.ptr);
        OnlineStream stream = new OnlineStream(s_ptr);
        return stream;
    }

    public float[] readWavFile(String filename) {
        // read data from the filename
        Object[] wavdata = readWave(filename);
        Object data = wavdata[0]; // data[0] is Int data, data[1] sample rate

        float[] floatData = (float[]) data;

        return floatData;
    }

    // load the libsherpa-onnx-jni.so lib
    public static String LoadSoLib() {

        Map<String, String> proMap = OnlineRecognizer.readProperties();

        // load libsherpa-onnx-jni.so lib from the path
        String SoPath = proMap.get("solibpath").trim();
        System.out.println("lib path=" + SoPath + "\n");
        System.load(SoPath);
        return SoPath; // return so path for stream init

    }

    //set model config file path
    public static void setCfgPath(String cfgPath) {

        OnlineRecognizer.cfgPath = cfgPath;
        String SoPath = OnlineRecognizer.LoadSoLib();
        OnlineStream.LoadSoLib(SoPath);
    }

    protected void finalize() throws Throwable {
        if (this.ptr == 0) return;
        DeleteOnlineRecognizer(this.ptr);
        this.ptr = 0;


    }

    // recognizer release, you'd better call it manually if not use anymore
    public void release() {
        if (this.ptr == 0) return;
        DeleteOnlineRecognizer(this.ptr);
        this.ptr = 0;
    }

    // stream release, you'd better call it manually if not use anymore
    public void releaseStream(OnlineStream s) {
        s.release();
    }
    // JNI interface libsherpa-onnx-jni.so

    private native Object[] readWave(String fileName);

    private native String GetResult(long ptr, long s_ptr);

    private native void DecodeStream(long ptr, long s_ptr);

    private native void DecodeStreams(long ptr, long[] ss_ptr);

    private native boolean IsReady(long ptr, long s_ptr);

    // first parameter keep for android asset_manager ANDROID_API__ >= 9
    private native long CreateOnlineRecognizer(Object asset, OnlineRecognizerConfig config);

    private native long CreateStream(long ptr);

    private native void DeleteOnlineRecognizer(long ptr);

    private native boolean IsEndpoint(long ptr, long s_ptr);

    private native void Reset(long ptr, long s_ptr);

}
