// Copyright 2024 Xiaomi Corporation

package com.k2fsa.sherpa.onnx;

public class VadModelConfig {
    private final SileroVadModelConfig sileroVadModelConfig;
    private final int sampleRate;
    private final int numThreads;
    private final boolean debug;
    private final String provider;

    private VadModelConfig(Builder builder) {
        this.sileroVadModelConfig = builder.sileroVadModelConfig;
        this.sampleRate = builder.sampleRate;
        this.numThreads = builder.numThreads;
        this.debug = builder.debug;
        this.provider = builder.provider;
    }

    public static Builder builder() {
        return new Builder();
    }

    public SileroVadModelConfig getSileroVadModelConfig() {
        return sileroVadModelConfig;
    }

    public int getSampleRate() {
        return sampleRate;
    }

    public int getNumThreads() {
        return numThreads;
    }

    public String getProvider() {
        return provider;
    }

    public boolean getDebug() {
        return debug;
    }

    public static class Builder {
        private SileroVadModelConfig sileroVadModelConfig = new SileroVadModelConfig.Builder().build();
        private int sampleRate = 16000;
        private int numThreads = 1;
        private boolean debug = true;
        private String provider = "cpu";

        public VadModelConfig build() {
            return new VadModelConfig(this);
        }

        public Builder setSileroVadModelConfig(SileroVadModelConfig sileroVadModelConfig) {
            this.sileroVadModelConfig = sileroVadModelConfig;
            return this;
        }

        public Builder setSampleRate(int sampleRate) {
            this.sampleRate = sampleRate;
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
