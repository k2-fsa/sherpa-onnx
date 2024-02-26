/*
 * // Copyright 2022-2023 by zhaoming
 */

package com.k2fsa.sherpa.onnx;

public class OnlineModelConfig {
    private final OnlineParaformerModelConfig paraformer;
    private final OnlineTransducerModelConfig transducer;
    private final OnlineZipformer2CtcModelConfig zipformer2Ctc;
    private final String tokens;
    private final int numThreads;
    private final boolean debug;
    private final String provider = "cpu";
    private String modelType = "";

    public OnlineModelConfig(
            String tokens,
            int numThreads,
            boolean debug,
            String modelType,
            OnlineParaformerModelConfig paraformer,
            OnlineTransducerModelConfig transducer,
            OnlineZipformer2CtcModelConfig zipformer2Ctc
            ) {

        this.tokens = tokens;
        this.numThreads = numThreads;
        this.debug = debug;
        this.modelType = modelType;
        this.paraformer = paraformer;
        this.transducer = transducer;
        this.zipformer2Ctc = zipformer2Ctc;
    }

    public OnlineParaformerModelConfig getParaformer() {
        return paraformer;
    }

    public OnlineTransducerModelConfig getTransducer() {
        return transducer;
    }

    public String getTokens() {
        return tokens;
    }

    public int getNumThreads() {
        return numThreads;
    }

    public boolean getDebug() {
        return debug;
    }
}
