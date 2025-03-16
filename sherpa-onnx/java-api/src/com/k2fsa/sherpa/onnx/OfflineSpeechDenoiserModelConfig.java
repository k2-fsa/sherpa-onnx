// Copyright 2025 Xiaomi Corporation
package com.k2fsa.sherpa.onnx;

public class OfflineSpeechDenoiserModelConfig {
    private final OfflineSpeechDenoiserGtcrnModelConfig gtcrn;
    private final int numThreads;
    private final boolean debug;
    private final String provider;

    private OfflineSpeechDenoiserModelConfig(Builder builder) {
        this.gtcrn = builder.gtcrn;
        this.numThreads = builder.numThreads;
        this.debug = builder.debug;
        this.provider = builder.provider;
    }

    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private OfflineSpeechDenoiserGtcrnModelConfig gtcrn = OfflineSpeechDenoiserGtcrnModelConfig.builder().build();
        private int numThreads = 1;
        private boolean debug = true;
        private String provider = "cpu";

        public OfflineSpeechDenoiserModelConfig build() {
            return new OfflineSpeechDenoiserModelConfig(this);
        }

        public Builder setGtcrn(OfflineSpeechDenoiserGtcrnModelConfig gtcrn) {
            this.gtcrn = gtcrn;
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
