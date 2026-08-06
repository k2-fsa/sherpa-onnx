// Copyright 2025 Xiaomi Corporation
package com.k2fsa.sherpa.onnx;

public class OfflineSpeechDenoiserDpdfNetModelConfig {
    private final String model;
    private final float attenuationLimitDb;

    private OfflineSpeechDenoiserDpdfNetModelConfig(Builder builder) {
        this.model = builder.model;
        this.attenuationLimitDb = builder.attenuationLimitDb;
    }

    public static Builder builder() {
        return new Builder();
    }

    public String getModel() {
        return model;
    }

    public float getAttenuationLimitDb() {
        return attenuationLimitDb;
    }

    public static class Builder {
        private String model = "";
        private float attenuationLimitDb = 0.0f;

        public OfflineSpeechDenoiserDpdfNetModelConfig build() {
            return new OfflineSpeechDenoiserDpdfNetModelConfig(this);
        }

        public Builder setModel(String model) {
            this.model = model;
            return this;
        }

        public Builder setAttenuationLimitDb(float value) {
            this.attenuationLimitDb = value;
            return this;
        }
    }
}
