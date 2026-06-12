// Copyright 2026 Matias Lin

package com.k2fsa.sherpa.onnx;

public class OfflineDiacritizationModelConfig {
    private final String cattEncoder;
    private final String cattDecoder;
    private final int numThreads;
    private final boolean debug;
    private final String provider;

    private OfflineDiacritizationModelConfig(Builder builder) {
        this.cattEncoder = builder.cattEncoder;
        this.cattDecoder = builder.cattDecoder;
        this.numThreads = builder.numThreads;
        this.debug = builder.debug;
        this.provider = builder.provider;
    }

    public static Builder builder() {
        return new Builder();
    }

    public String getCattEncoder() {
        return cattEncoder;
    }

    public String getCattDecoder() {
        return cattDecoder;
    }

    public static class Builder {
        private String cattEncoder = "";
        private String cattDecoder = "";
        private int numThreads = 1;
        private boolean debug = true;
        private String provider = "cpu";

        public OfflineDiacritizationModelConfig build() {
            return new OfflineDiacritizationModelConfig(this);
        }

        public Builder setCattEncoder(String cattEncoder) {
            this.cattEncoder = cattEncoder;
            return this;
        }

        public Builder setCattDecoder(String cattDecoder) {
            this.cattDecoder = cattDecoder;
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
