package com.k2fsa.sherpa.onnx;

public class OfflineMedAsrCtcModelConfig {
    private final String model;

    private OfflineMedAsrCtcModelConfig(Builder builder) {
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

        public OfflineMedAsrCtcModelConfig build() {
            return new OfflineMedAsrCtcModelConfig(this);
        }

        public Builder setModel(String model) {
            this.model = model;
            return this;
        }
    }
}
