// Copyright 2024 Xiaomi Corporation
package com.k2fsa.sherpa.onnx;

public class OnlineNeMoCtcModelConfig {
    private final String model;

    private OnlineNeMoCtcModelConfig(Builder builder) {
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

        public OnlineNeMoCtcModelConfig build() {
            return new OnlineNeMoCtcModelConfig(this);
        }

        public Builder setModel(String model) {
            this.model = model;
            return this;
        }
    }

}
