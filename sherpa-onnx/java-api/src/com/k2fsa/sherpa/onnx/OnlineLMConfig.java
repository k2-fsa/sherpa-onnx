// Copyright 2022-2023 by zhaoming
// Copyright 2024 Xiaomi Corporation

package com.k2fsa.sherpa.onnx;

public class OnlineLMConfig {

    private final String model;
    private final float scale;

    private OnlineLMConfig(Builder builder) {
        this.model = builder.model;
        this.scale = builder.scale;
    }

    public static Builder builder() {
        return new Builder();
    }

    public String getModel() {
        return model;
    }

    public float getScale() {
        return scale;
    }

    public static class Builder {
        private String model = "";
        private float scale = 1.0f;

        public OnlineLMConfig build() {
            return new OnlineLMConfig(this);
        }

        public Builder setModel(String model) {
            this.model = model;
            return this;
        }

        public Builder setScale(float scale) {
            this.scale = scale;
            return this;
        }
    }
}