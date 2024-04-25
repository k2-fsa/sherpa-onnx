// Copyright 2024 Xiaomi Corporation
package com.k2fsa.sherpa.onnx;

public class OnlineCtcFstDecoderConfig {
    private final String graph;
    private final int maxActive;

    private OnlineCtcFstDecoderConfig(Builder builder) {
        this.graph = builder.graph;
        this.maxActive = builder.maxActive;
    }

    public static Builder builder() {
        return new Builder();
    }

    public String getGraph() {
        return graph;
    }

    public float getMaxActive() {
        return maxActive;
    }

    public static class Builder {
        private String graph = "";
        private int maxActive = 3000;

        public OnlineCtcFstDecoderConfig build() {
            return new OnlineCtcFstDecoderConfig(this);
        }

        public Builder setGraph(String model) {
            this.graph = graph;
            return this;
        }

        public Builder setMaxActive(int maxActive) {
            this.maxActive = maxActive;
            return this;
        }
    }
}