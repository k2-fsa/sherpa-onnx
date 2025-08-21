// Copyright 2024 Xiaomi Corporation

package com.k2fsa.sherpa.onnx;

public class OnlinePunctuationConfig {
    private final OnlinePunctuationModelConfig model;

    private OnlinePunctuationConfig(Builder builder) {
        this.model = builder.model;
    }

    public static Builder builder() {
        return new Builder();
    }

    public OnlinePunctuationModelConfig getModel() {
        return model;
    }


    public static class Builder {
        private OnlinePunctuationModelConfig model = OnlinePunctuationModelConfig.builder().build();

        public OnlinePunctuationConfig build() {
            return new OnlinePunctuationConfig(this);
        }

        public Builder setModel(OnlinePunctuationModelConfig model) {
            this.model = model;
            return this;
        }
    }
}
