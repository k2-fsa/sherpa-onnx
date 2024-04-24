// Copyright 2024 Xiaomi Corporation
package com.k2fsa.sherpa.onnx;

public class OfflineParaformerModelConfig {
    private final String model;

    private OfflineParaformerModelConfig(Builder builder) {
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

        public OfflineParaformerModelConfig build() {
            return new OfflineParaformerModelConfig(this);
        }

        public Builder setModel(String model) {
            this.model = model;
            return this;
        }
    }
}
