// Copyright 2025 Xiaomi Corporation
package com.k2fsa.sherpa.onnx;

public class OfflineSpeechDenoiserGtcrnModelConfig {
    private final String model;

    private OfflineSpeechDenoiserGtcrnModelConfig(Builder builder) {
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

        public OfflineSpeechDenoiserGtcrnModelConfig build() {
            return new OfflineSpeechDenoiserGtcrnModelConfig(this);
        }

        public Builder setModel(String model) {
            this.model = model;
            return this;
        }
    }
}
