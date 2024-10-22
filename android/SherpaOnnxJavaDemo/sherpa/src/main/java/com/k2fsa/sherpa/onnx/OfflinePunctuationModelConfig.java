// Copyright 2024 Xiaomi Corporation

package com.k2fsa.sherpa.onnx;

public class OfflinePunctuationModelConfig {
    private final String ctTransformer;
    private final int numThreads;
    private final boolean debug;
    private final String provider;

    private OfflinePunctuationModelConfig(Builder builder) {
        this.ctTransformer = builder.ctTransformer;
        this.numThreads = builder.numThreads;
        this.debug = builder.debug;
        this.provider = builder.provider;
    }

    public static Builder builder() {
        return new Builder();
    }

    public String getCtTransformer() {
        return ctTransformer;
    }

    public static class Builder {
        private String ctTransformer = "";
        private int numThreads = 1;
        private boolean debug = true;
        private String provider = "cpu";

        public OfflinePunctuationModelConfig build() {
            return new OfflinePunctuationModelConfig(this);
        }

        public Builder setCtTransformer(String ctTransformer) {
            this.ctTransformer = ctTransformer;
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
