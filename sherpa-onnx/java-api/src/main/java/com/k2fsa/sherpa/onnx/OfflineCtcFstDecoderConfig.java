// Copyright 2026 Marcin Baszczewski

package com.k2fsa.sherpa.onnx;

public class OfflineCtcFstDecoderConfig {
    private final String graph;
    private final int maxActive;

    private OfflineCtcFstDecoderConfig(Builder builder) {
        this.graph = builder.graph;
        this.maxActive = builder.maxActive;
    }

    public static Builder builder() {
        return new Builder();
    }

    public String getGraph() {
        return graph;
    }

    public int getMaxActive() {
        return maxActive;
    }

    public static class Builder {
        private String graph = "";
        private int maxActive = 3000;

        public OfflineCtcFstDecoderConfig build() {
            return new OfflineCtcFstDecoderConfig(this);
        }

        public Builder setGraph(String graph) {
            this.graph = graph;
            return this;
        }

        public Builder setMaxActive(int maxActive) {
            this.maxActive = maxActive;
            return this;
        }
    }
}
