// Copyright 2024 Xiaomi Corporation

package com.k2fsa.sherpa.onnx;

public class SpokenLanguageIdentificationConfig {
    private final SpokenLanguageIdentificationWhisperConfig whisper;
    private final int numThreads;
    private final boolean debug;
    private final String provider;

    private SpokenLanguageIdentificationConfig(Builder builder) {
        this.whisper = builder.whisper;
        this.numThreads = builder.numThreads;
        this.debug = builder.debug;
        this.provider = builder.provider;
    }

    public static Builder builder() {
        return new Builder();
    }

    public SpokenLanguageIdentificationWhisperConfig getWhisper() {
        return whisper;
    }

    public static class Builder {
        private SpokenLanguageIdentificationWhisperConfig whisper = SpokenLanguageIdentificationWhisperConfig.builder().build();
        private int numThreads = 1;
        private boolean debug = true;
        private String provider = "cpu";

        public SpokenLanguageIdentificationConfig build() {
            return new SpokenLanguageIdentificationConfig(this);
        }

        public Builder setWhisper(SpokenLanguageIdentificationWhisperConfig whisper) {
            this.whisper = whisper;
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
