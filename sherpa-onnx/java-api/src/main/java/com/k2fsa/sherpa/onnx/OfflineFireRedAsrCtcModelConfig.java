package com.k2fsa.sherpa.onnx;

public class OfflineFireRedAsrCtcModelConfig {
    private final String model;

    private OfflineFireRedAsrCtcModelConfig(Builder builder) {
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

        public OfflineFireRedAsrCtcModelConfig build() {
            return new OfflineFireRedAsrCtcModelConfig(this);
        }

        public Builder setModel(String model) {
            this.model = model;
            return this;
        }
    }
}
