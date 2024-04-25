// Copyright 2024 Xiaomi Corporation

package com.k2fsa.sherpa.onnx;

public class OfflineTtsModelConfig {
    private final OfflineTtsVitsModelConfig vits;
    private final int numThreads;
    private final boolean debug;
    private final String provider;

    private OfflineTtsModelConfig(Builder builder) {
        this.vits = builder.vits;
        this.numThreads = builder.numThreads;
        this.debug = builder.debug;
        this.provider = builder.provider;
    }

    public static Builder builder() {
        return new Builder();
    }

    public OfflineTtsVitsModelConfig getVits() {
        return vits;
    }

    public static class Builder {
        private OfflineTtsVitsModelConfig vits = OfflineTtsVitsModelConfig.builder().build();
        private int numThreads = 1;
        private boolean debug = true;
        private String provider = "cpu";

        public OfflineTtsModelConfig build() {
            return new OfflineTtsModelConfig(this);
        }

        public Builder setVits(OfflineTtsVitsModelConfig vits) {
            this.vits = vits;
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
