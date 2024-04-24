// Copyright 2022-2023 by zhaoming
// Copyright 2024 Xiaomi Corporation

package com.k2fsa.sherpa.onnx;

public class OnlineModelConfig {
    private final OnlineTransducerModelConfig transducer;
    private final OnlineParaformerModelConfig paraformer;
    private final OnlineZipformer2CtcModelConfig zipformer2Ctc;
    private final String tokens;
    private final int numThreads;
    private final boolean debug;
    private final String provider;
    private final String modelType;
    private OnlineModelConfig(Builder builder) {
        this.transducer = builder.transducer;
        this.paraformer = builder.paraformer;
        this.zipformer2Ctc = builder.zipformer2Ctc;
        this.tokens = builder.tokens;
        this.numThreads = builder.numThreads;
        this.debug = builder.debug;
        this.provider = builder.provider;
        this.modelType = builder.modelType;
    }

    public static Builder builder() {
        return new Builder();
    }

    public OnlineParaformerModelConfig getParaformer() {
        return paraformer;
    }

    public OnlineTransducerModelConfig getTransducer() {
        return transducer;
    }

    public OnlineZipformer2CtcModelConfig getZipformer2Ctc() {
        return zipformer2Ctc;
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

    public String getProvider() {
        return provider;
    }

    public String getModelType() {
        return modelType;
    }

    public static class Builder {
        private OnlineParaformerModelConfig paraformer = OnlineParaformerModelConfig.builder().build();
        private OnlineTransducerModelConfig transducer = OnlineTransducerModelConfig.builder().build();
        private OnlineZipformer2CtcModelConfig zipformer2Ctc = OnlineZipformer2CtcModelConfig.builder().build();
        private String tokens = "";
        private int numThreads = 1;
        private boolean debug = true;
        private String provider = "cpu";
        private String modelType = "";

        public OnlineModelConfig build() {
            return new OnlineModelConfig(this);
        }

        public Builder setTransducer(OnlineTransducerModelConfig transducer) {
            this.transducer = transducer;
            return this;
        }

        public Builder setParaformer(OnlineParaformerModelConfig paraformer) {
            this.paraformer = paraformer;
            return this;
        }

        public Builder setZipformer2Ctc(OnlineZipformer2CtcModelConfig zipformer2Ctc) {
            this.zipformer2Ctc = zipformer2Ctc;
            return this;
        }

        public Builder setTokens(String tokens) {
            this.tokens = tokens;
            return this;
        }

        public Builder setNumThreads(int numThreads) {
            this.numThreads = numThreads;
            return this;
        }

        public Builder setDebug(boolean debug) {
            this.debug = debug;
            return this;
        }

        public Builder setProvider(String provider) {
            this.provider = provider;
            return this;
        }

        public Builder setModelType(String modelType) {
            this.modelType = modelType;
            return this;
        }
    }
}
