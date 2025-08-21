// Copyright 2024 Xiaomi Corporation

package com.k2fsa.sherpa.onnx;

public class SpokenLanguageIdentificationWhisperConfig {
    private final String encoder;
    private final String decoder;
    private final int tailPaddings;

    private SpokenLanguageIdentificationWhisperConfig(Builder builder) {
        this.encoder = builder.encoder;
        this.decoder = builder.decoder;
        this.tailPaddings = builder.tailPaddings;
    }

    public static Builder builder() {
        return new Builder();
    }

    public String getEncoder() {
        return encoder;
    }

    public String getDecoder() {
        return decoder;
    }

    public int getTailPaddings() {
        return tailPaddings;
    }

    public static class Builder {
        private String encoder = "";
        private String decoder = "";
        private int tailPaddings = 1000; // number of frames to pad

        public SpokenLanguageIdentificationWhisperConfig build() {
            return new SpokenLanguageIdentificationWhisperConfig(this);
        }

        public Builder setEncoder(String encoder) {
            this.encoder = encoder;
            return this;
        }

        public Builder setDecoder(String decoder) {
            this.decoder = decoder;
            return this;
        }

        public Builder setTailPaddings(int tailPaddings) {
            this.tailPaddings = tailPaddings;
            return this;
        }
    }
}
