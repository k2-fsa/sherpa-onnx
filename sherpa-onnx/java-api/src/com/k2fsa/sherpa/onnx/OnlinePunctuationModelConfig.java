// Copyright 2024 Xiaomi Corporation

package com.k2fsa.sherpa.onnx;

public class OnlinePunctuationModelConfig {
    private final String cnnBilstm;
    private final String bpeVocab;
    private final int numThreads;
    private final boolean debug;
    private final String provider;

    private OnlinePunctuationModelConfig(Builder builder) {
        this.cnnBilstm = builder.cnnBilstm;
        this.bpeVocab = builder.bpeVocab;
        this.numThreads = builder.numThreads;
        this.debug = builder.debug;
        this.provider = builder.provider;
    }

    public static Builder builder() {
        return new Builder();
    }

    public String getCnnBilstm() {
        return cnnBilstm;
    }

    public String getBpeVocab() {
        return bpeVocab;
    }

    public static class Builder {
        private String cnnBilstm = "";
        private String bpeVocab = "";
        private int numThreads = 1;
        private boolean debug = true;
        private String provider = "cpu";

        public OnlinePunctuationModelConfig build() {
            return new OnlinePunctuationModelConfig(this);
        }

        public Builder setCnnBilstm(String cnnBilstm) {
            this.cnnBilstm = cnnBilstm;
            return this;
        }

        public Builder setBpeVocab(String bpeVocab) {
            this.bpeVocab = bpeVocab;
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
    }
}
