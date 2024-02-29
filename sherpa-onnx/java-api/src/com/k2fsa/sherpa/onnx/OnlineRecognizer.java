/*
 * // Copyright 2022-2023 by zhaoming
 * // the online recognizer for sherpa-onnx, it can load config from a file
 * // or by argument
 */
/*
usage example:

    String cfgpath=appdir+"/modelconfig.cfg";
	OnlineRecognizer.setSoPath(soPath);   //set so lib path

    OnlineRecognizer rcgOjb = new OnlineRecognizer();   //create a recognizer
    rcgOjb = new OnlineRecognizer(cfgFile);    //set model config file
    CreateStream streamObj=rcgOjb.CreateStream();       //create a stream for read wav data
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

*/
package com.k2fsa.sherpa.onnx;

import java.io.BufferedInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStream;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.Map;
import java.util.Properties;

public class OnlineRecognizer {
    private long ptr = 0; // this is the asr engine ptrss

    private int sampleRate = 16000;

    // load config file for OnlineRecognizer
    public OnlineRecognizer(String modelCfgPath) {
        Map<String, String> proMap = this.readProperties(modelCfgPath);
        try {
            int sampleRate = Integer.parseInt(proMap.getOrDefault("sample_rate", "16000").trim());
            this.sampleRate = sampleRate;
            EndpointRule rule1 =
                    new EndpointRule(
                            false,
                            Float.parseFloat(proMap.getOrDefault("rule1_min_trailing_silence", "2.4").trim()),
                            0.0F);
            EndpointRule rule2 =
                    new EndpointRule(
                            true,
                            Float.parseFloat(proMap.getOrDefault("rule2_min_trailing_silence", "1.2").trim()),
                            0.0F);
            EndpointRule rule3 =
                    new EndpointRule(
                            false,
                            0.0F,
                            Float.parseFloat(proMap.getOrDefault("rule3_min_utterance_length", "20").trim()));
            EndpointConfig endCfg = new EndpointConfig(rule1, rule2, rule3);

            OnlineParaformerModelConfig modelParaCfg =
                    new OnlineParaformerModelConfig(
                            proMap.getOrDefault("encoder", "").trim(), proMap.getOrDefault("decoder", "").trim());
            OnlineTransducerModelConfig modelTranCfg =
                    new OnlineTransducerModelConfig(
                            proMap.getOrDefault("encoder", "").trim(),
                            proMap.getOrDefault("decoder", "").trim(),
                            proMap.getOrDefault("joiner", "").trim());
            OnlineZipformer2CtcModelConfig zipformer2CtcConfig = new OnlineZipformer2CtcModelConfig("");
            OnlineModelConfig modelCfg =
                    new OnlineModelConfig(
                            proMap.getOrDefault("tokens", "").trim(),
                            Integer.parseInt(proMap.getOrDefault("num_threads", "4").trim()),
                            false,
                            proMap.getOrDefault("model_type", "zipformer").trim(),
                            modelParaCfg,
                            modelTranCfg, zipformer2CtcConfig);
            FeatureConfig featConfig =
                    new FeatureConfig(
                            sampleRate, Integer.parseInt(proMap.getOrDefault("feature_dim", "80").trim()));
            OnlineLMConfig onlineLmConfig =
                    new OnlineLMConfig(
                            proMap.getOrDefault("lm_model", "").trim(),
                            Float.parseFloat(proMap.getOrDefault("lm_scale", "0.5").trim()));

            OnlineRecognizerConfig rcgCfg =
                    new OnlineRecognizerConfig(
                            featConfig,
                            modelCfg,
                            endCfg,
                            onlineLmConfig,
                            Boolean.parseBoolean(proMap.getOrDefault("enable_endpoint_detection", "true").trim()),
                            proMap.getOrDefault("decoding_method", "modified_beam_search").trim(),
                            Integer.parseInt(proMap.getOrDefault("max_active_paths", "4").trim()),
                            proMap.getOrDefault("hotwords_file", "").trim(),
                            Float.parseFloat(proMap.getOrDefault("hotwords_score", "1.5").trim()));
            // create a new Recognizer, first parameter kept for android asset_manager ANDROID_API__ >= 9
            this.ptr = createOnlineRecognizer(new Object(), rcgCfg);

        } catch (Exception e) {
            System.err.println(e);
        }
    }

    // use for android asset_manager ANDROID_API__ >= 9
    public OnlineRecognizer(Object assetManager, String modelCfgPath) {
        Map<String, String> proMap = this.readProperties(modelCfgPath);
        try {
            int sampleRate = Integer.parseInt(proMap.getOrDefault("sample_rate", "16000").trim());
            this.sampleRate = sampleRate;
            EndpointRule rule1 =
                    new EndpointRule(
                            false,
                            Float.parseFloat(proMap.getOrDefault("rule1_min_trailing_silence", "2.4").trim()),
                            0.0F);
            EndpointRule rule2 =
                    new EndpointRule(
                            true,
                            Float.parseFloat(proMap.getOrDefault("rule2_min_trailing_silence", "1.2").trim()),
                            0.0F);
            EndpointRule rule3 =
                    new EndpointRule(
                            false,
                            0.0F,
                            Float.parseFloat(proMap.getOrDefault("rule3_min_utterance_length", "20").trim()));
            EndpointConfig endCfg = new EndpointConfig(rule1, rule2, rule3);
            OnlineParaformerModelConfig modelParaCfg =
                    new OnlineParaformerModelConfig(
                            proMap.getOrDefault("encoder", "").trim(), proMap.getOrDefault("decoder", "").trim());
            OnlineTransducerModelConfig modelTranCfg =
                    new OnlineTransducerModelConfig(
                            proMap.getOrDefault("encoder", "").trim(),
                            proMap.getOrDefault("decoder", "").trim(),
                            proMap.getOrDefault("joiner", "").trim());
            OnlineZipformer2CtcModelConfig zipformer2CtcConfig = new OnlineZipformer2CtcModelConfig("");

            OnlineModelConfig modelCfg =
                    new OnlineModelConfig(
                            proMap.getOrDefault("tokens", "").trim(),
                            Integer.parseInt(proMap.getOrDefault("num_threads", "4").trim()),
                            false,
                            proMap.getOrDefault("model_type", "zipformer").trim(),
                            modelParaCfg,
                            modelTranCfg, zipformer2CtcConfig);
            FeatureConfig featConfig =
                    new FeatureConfig(
                            sampleRate, Integer.parseInt(proMap.getOrDefault("feature_dim", "80").trim()));

            OnlineLMConfig onlineLmConfig =
                    new OnlineLMConfig(
                            proMap.getOrDefault("lm_model", "").trim(),
                            Float.parseFloat(proMap.getOrDefault("lm_scale", "0.5").trim()));

            OnlineRecognizerConfig rcgCfg =
                    new OnlineRecognizerConfig(
                            featConfig,
                            modelCfg,
                            endCfg,
                            onlineLmConfig,
                            Boolean.parseBoolean(proMap.getOrDefault("enable_endpoint_detection", "true").trim()),
                            proMap.getOrDefault("decoding_method", "modified_beam_search").trim(),
                            Integer.parseInt(proMap.getOrDefault("max_active_paths", "4").trim()),
                            proMap.getOrDefault("hotwords_file", "").trim(),
                            Float.parseFloat(proMap.getOrDefault("hotwords_score", "1.5").trim()));
            // create a new Recognizer, first parameter kept for android asset_manager ANDROID_API__ >= 9
            this.ptr = createOnlineRecognizer(assetManager, rcgCfg);

        } catch (Exception e) {
            System.err.println(e);
        }
    }

    // set onlineRecognizer by parameter
    public OnlineRecognizer(
            String tokens,
            String encoder,
            String decoder,
            String joiner,
            int numThreads,
            int sampleRate,
            int featureDim,
            boolean enableEndpointDetection,
            float rule1MinTrailingSilence,
            float rule2MinTrailingSilence,
            float rule3MinUtteranceLength,
            String decodingMethod,
            String lm_model,
            float lm_scale,
            int maxActivePaths,
            String hotwordsFile,
            float hotwordsScore,
            String modelType) {
        this.sampleRate = sampleRate;
        EndpointRule rule1 = new EndpointRule(false, rule1MinTrailingSilence, 0.0F);
        EndpointRule rule2 = new EndpointRule(true, rule2MinTrailingSilence, 0.0F);
        EndpointRule rule3 = new EndpointRule(false, 0.0F, rule3MinUtteranceLength);
        EndpointConfig endCfg = new EndpointConfig(rule1, rule2, rule3);
        OnlineParaformerModelConfig modelParaCfg = new OnlineParaformerModelConfig(encoder, decoder);
        OnlineTransducerModelConfig modelTranCfg =
                new OnlineTransducerModelConfig(encoder, decoder, joiner);
        OnlineZipformer2CtcModelConfig zipformer2CtcConfig = new OnlineZipformer2CtcModelConfig("");
        OnlineModelConfig modelCfg =
                new OnlineModelConfig(tokens, numThreads, false, modelType, modelParaCfg, modelTranCfg, zipformer2CtcConfig);
        FeatureConfig featConfig = new FeatureConfig(sampleRate, featureDim);
        OnlineLMConfig onlineLmConfig = new OnlineLMConfig(lm_model, lm_scale);
        OnlineRecognizerConfig rcgCfg =
                new OnlineRecognizerConfig(
                        featConfig,
                        modelCfg,
                        endCfg,
                        onlineLmConfig,
                        enableEndpointDetection,
                        decodingMethod,
                        maxActivePaths,
                        hotwordsFile,
                        hotwordsScore);
        // create a new Recognizer, first parameter kept for android asset_manager ANDROID_API__ >= 9
        this.ptr = createOnlineRecognizer(new Object(), rcgCfg);
    }

    public static float[] readWavFile(String fileName) {
        // read data from the filename
        Object[] wavdata = readWave(fileName);
        Object data = wavdata[0]; // data[0] is float data, data[1] sample rate

        float[] floatData = (float[]) data;

        return floatData;
    }

    // load the libsherpa-onnx-jni.so lib
    public static void loadSoLib(String soPath) {
        // load libsherpa-onnx-jni.so lib from the path

        System.out.println("so lib path=" + soPath + "\n");
        System.load(soPath.trim());
        System.out.println("load so lib succeed\n");
    }

    public static void setSoPath(String soPath) {
        OnlineRecognizer.loadSoLib(soPath);
        OnlineStream.loadSoLib(soPath);
    }

    private static native Object[] readWave(String fileName); // static

    private Map<String, String> readProperties(String modelCfgPath) {
        // read and parse config file
        Properties props = new Properties();
        Map<String, String> proMap = new HashMap<>();
        try {
            File file = new File(modelCfgPath);
            if (!file.exists()) {
                System.out.println("model cfg file not exists!");
                System.exit(0);
            }
            InputStream in = new BufferedInputStream(new FileInputStream(modelCfgPath));
            props.load(in);
            Enumeration en = props.propertyNames();
            while (en.hasMoreElements()) {
                String key = (String) en.nextElement();
                String Property = props.getProperty(key);
                proMap.put(key, Property);
            }

        } catch (Exception e) {
            e.printStackTrace();
        }
        return proMap;
    }

    public void decodeStream(OnlineStream s) throws Exception {
        if (this.ptr == 0) throw new Exception("null exception for recognizer ptr");
        long streamPtr = s.getPtr();
        if (streamPtr == 0) throw new Exception("null exception for stream ptr");
        // when feeded samples to engine, call DecodeStream to let it process
        decodeStream(this.ptr, streamPtr);
    }

    public void decodeStreams(OnlineStream[] ssOjb) throws Exception {
        if (this.ptr == 0) throw new Exception("null exception for recognizer ptr");
        // decode for multiple streams
        long[] ss = new long[ssOjb.length];
        for (int i = 0; i < ssOjb.length; i++) {
            ss[i] = ssOjb[i].getPtr();
            if (ss[i] == 0) throw new Exception("null exception for stream ptr");
        }
        decodeStreams(this.ptr, ss);
    }

    public boolean isReady(OnlineStream s) throws Exception {
        // whether the engine is ready for decode
        if (this.ptr == 0) throw new Exception("null exception for recognizer ptr");
        long streamPtr = s.getPtr();
        if (streamPtr == 0) throw new Exception("null exception for stream ptr");
        return isReady(this.ptr, streamPtr);
    }

    public String getResult(OnlineStream s) throws Exception {
        // get text from the engine
        if (this.ptr == 0) throw new Exception("null exception for recognizer ptr");
        long streamPtr = s.getPtr();
        if (streamPtr == 0) throw new Exception("null exception for stream ptr");
        return getResult(this.ptr, streamPtr);
    }

    public boolean isEndpoint(OnlineStream s) throws Exception {
        if (this.ptr == 0) throw new Exception("null exception for recognizer ptr");
        long streamPtr = s.getPtr();
        if (streamPtr == 0) throw new Exception("null exception for stream ptr");
        return isEndpoint(this.ptr, streamPtr);
    }

    public void reSet(OnlineStream s) throws Exception {
        if (this.ptr == 0) throw new Exception("null exception for recognizer ptr");
        long streamPtr = s.getPtr();
        if (streamPtr == 0) throw new Exception("null exception for stream ptr");
        reSet(this.ptr, streamPtr);
    }

    public OnlineStream createStream() throws Exception {
        // create one stream for data to feed in
        if (this.ptr == 0) throw new Exception("null exception for recognizer ptr");
        long streamPtr = createStream(this.ptr);
        OnlineStream stream = new OnlineStream(streamPtr, this.sampleRate);
        return stream;
    }

    protected void finalize() throws Throwable {
        release();
    }

    // recognizer release, you'd better call it manually if not use anymore
    public void release() {
        if (this.ptr == 0) return;
        deleteOnlineRecognizer(this.ptr);
        this.ptr = 0;
    }

    // JNI interface libsherpa-onnx-jni.so

    // stream release, you'd better call it manually if not use anymore
    public void releaseStream(OnlineStream s) {
        s.release();
    }

    private native String getResult(long ptr, long streamPtr);

    private native void decodeStream(long ptr, long streamPtr);

    private native void decodeStreams(long ptr, long[] ssPtr);

    private native boolean isReady(long ptr, long streamPtr);

    // first parameter keep for android asset_manager ANDROID_API__ >= 9
    private native long createOnlineRecognizer(Object asset, OnlineRecognizerConfig config);

    private native long createStream(long ptr);

    private native void deleteOnlineRecognizer(long ptr);

    private native boolean isEndpoint(long ptr, long streamPtr);

    private native void reSet(long ptr, long streamPtr);
}
