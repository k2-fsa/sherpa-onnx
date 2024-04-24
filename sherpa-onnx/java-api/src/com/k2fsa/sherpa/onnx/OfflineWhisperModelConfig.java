// Copyright 2024 Xiaomi Corporation
package com.k2fsa.sherpa.onnx;

public class OfflineWhisperModelConfig {
    private final String encoder;
    private final String decoder;
    private final String language;
    private final String task;
    private final int tailPaddings;

    private OfflineWhisperModelConfig(Builder builder) {
        this.encoder = builder.encoder;
        this.decoder = builder.decoder;
        this.language = builder.language;
        this.task = builder.task;
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

    public String getLanguage() {
        return language;
    }

    public String getTask() {
        return task;
    }

    public int getTailPaddings() {
        return tailPaddings;
    }

    public static class Builder {
        private String encoder = "";
        private String decoder = "";
        private String language = "en"; // used only with multilingual models
        private String task = "transcribe"; // used only with multilingual models

        private int tailPaddings = 1000; // number of frames to pad

        public OfflineWhisperModelConfig build() {
            return new OfflineWhisperModelConfig(this);
        }

        public Builder setEncoder(String encoder) {
            this.encoder = encoder;
            return this;
        }

        public Builder setDecoder(String decoder) {
            this.decoder = decoder;
            return this;
        }

        public Builder setLanguage(String language) {
            this.language = language;
            return this;
        }

        public Builder setTask(String task) {
            this.task = task;
            return this;
        }

        public Builder setTailPaddings(int tailPaddings) {
            this.tailPaddings = tailPaddings;
            return this;
        }
    }
}
