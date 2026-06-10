// Copyright 2026 Matias Lin

package com.k2fsa.sherpa.onnx;

public class OfflineDiacritizationConfig {
    private final OfflineDiacritizationModelConfig model;

    private OfflineDiacritizationConfig(Builder builder) {
        this.model = builder.model;
    }

    public static Builder builder() {
        return new Builder();
    }

    public OfflineDiacritizationModelConfig getModel() {
        return model;
    }

    public static class Builder {
        private OfflineDiacritizationModelConfig model = OfflineDiacritizationModelConfig.builder().build();

        public OfflineDiacritizationConfig build() {
            return new OfflineDiacritizationConfig(this);
        }

        public Builder setModel(OfflineDiacritizationModelConfig model) {
            this.model = model;
            return this;
        }
    }
}
