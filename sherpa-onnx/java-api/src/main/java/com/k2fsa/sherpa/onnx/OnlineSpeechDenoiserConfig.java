// Copyright 2026 Xiaomi Corporation
package com.k2fsa.sherpa.onnx;

public class OnlineSpeechDenoiserConfig {
    private final OfflineSpeechDenoiserModelConfig model;

    private OnlineSpeechDenoiserConfig(Builder builder) {
        this.model = builder.model;
    }

    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private OfflineSpeechDenoiserModelConfig model = OfflineSpeechDenoiserModelConfig.builder().build();

        public OnlineSpeechDenoiserConfig build() {
            return new OnlineSpeechDenoiserConfig(this);
        }

        public Builder setModel(OfflineSpeechDenoiserModelConfig model) {
            this.model = model;
            return this;
        }
    }
}
