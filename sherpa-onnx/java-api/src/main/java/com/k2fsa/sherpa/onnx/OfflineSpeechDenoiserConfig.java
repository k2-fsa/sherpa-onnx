// Copyright 2025 Xiaomi Corporation
package com.k2fsa.sherpa.onnx;

public class OfflineSpeechDenoiserConfig {
    private final OfflineSpeechDenoiserModelConfig model;
    private final float dpdfnetAttenuationLimitDb;

    private OfflineSpeechDenoiserConfig(OfflineSpeechDenoiserConfig.Builder builder) {
        this.model = builder.model;
        this.dpdfnetAttenuationLimitDb = builder.dpdfnetAttenuationLimitDb;
    }

    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private OfflineSpeechDenoiserModelConfig model = OfflineSpeechDenoiserModelConfig.builder().build();
        private float dpdfnetAttenuationLimitDb = 0.0f;

        public OfflineSpeechDenoiserConfig build() {
            return new OfflineSpeechDenoiserConfig(this);
        }

        public Builder setModel(OfflineSpeechDenoiserModelConfig model) {
            this.model = model;
            return this;
        }

        public Builder setDpdfnetAttenuationLimitDb(float value) {
            this.dpdfnetAttenuationLimitDb = value;
            return this;
        }
    }
}
