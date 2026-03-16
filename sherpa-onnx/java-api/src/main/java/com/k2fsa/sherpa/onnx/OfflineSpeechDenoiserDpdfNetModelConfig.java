// Copyright 2025 Xiaomi Corporation
package com.k2fsa.sherpa.onnx;

public class OfflineSpeechDenoiserDpdfNetModelConfig {
    private final String model;

    private OfflineSpeechDenoiserDpdfNetModelConfig(Builder builder) {
        this.model = builder.model;
    }

    public static Builder builder() {
        return new Builder();
    }

    public String getModel() {
        return model;
    }

    public static class Builder {
        private String model = "";

        public OfflineSpeechDenoiserDpdfNetModelConfig build() {
            return new OfflineSpeechDenoiserDpdfNetModelConfig(this);
        }

        public Builder setModel(String model) {
            this.model = model;
            return this;
        }
    }
}
