// Copyright 2024 Xiaomi Corporation

package com.k2fsa.sherpa.onnx;

public class OfflinePunctuationConfig {
    private final OfflinePunctuationModelConfig model;

    private OfflinePunctuationConfig(Builder builder) {
        this.model = builder.model;
    }

    public static Builder builder() {
        return new Builder();
    }

    public OfflinePunctuationModelConfig getModel() {
        return model;
    }


    public static class Builder {
        private OfflinePunctuationModelConfig model = OfflinePunctuationModelConfig.builder().build();

        public OfflinePunctuationConfig build() {
            return new OfflinePunctuationConfig(this);
        }

        public Builder setModel(OfflinePunctuationModelConfig model) {
            this.model = model;
            return this;
        }
    }
}
