// Copyright 2024 Xiaomi Corporation
package com.k2fsa.sherpa.onnx;

public class OfflineModelConfig {
    private final OfflineTransducerModelConfig transducer;
    private final OfflineParaformerModelConfig paraformer;
    private final OfflineWhisperModelConfig whisper;
    private final String tokens;
    private final int numThreads;
    private final boolean debug;
    private final String provider;

    private final String modelType;

    private OfflineModelConfig(Builder builder) {
        this.transducer = builder.transducer;
        this.paraformer = builder.paraformer;
        this.whisper = builder.whisper;
        this.tokens = builder.tokens;
        this.numThreads = builder.numThreads;
        this.debug = builder.debug;
        this.provider = builder.provider;
        this.modelType = builder.modelType;
    }

    public static Builder builder() {
        return new Builder();
    }

    public OfflineParaformerModelConfig getParaformer() {
        return paraformer;
    }

    public OfflineTransducerModelConfig getTransducer() {
        return transducer;
    }

    public OfflineWhisperModelConfig getZipformer2Ctc() {
        return whisper;
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
        private OfflineParaformerModelConfig paraformer = OfflineParaformerModelConfig.builder().build();
        private OfflineTransducerModelConfig transducer = OfflineTransducerModelConfig.builder().build();
        private OfflineWhisperModelConfig whisper = OfflineWhisperModelConfig.builder().build();
        private String tokens = "";
        private int numThreads = 1;
        private boolean debug = true;
        private String provider = "cpu";
        private String modelType = "";

        public OfflineModelConfig build() {
            return new OfflineModelConfig(this);
        }

        public Builder setTransducer(OfflineTransducerModelConfig transducer) {
            this.transducer = transducer;
            return this;
        }

        public Builder setParaformer(OfflineParaformerModelConfig paraformer) {
            this.paraformer = paraformer;
            return this;
        }

        public Builder setWhisper(OfflineWhisperModelConfig whisper) {
            this.whisper = whisper;
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