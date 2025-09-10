package com.k2fsa.sherpa.onnx;

public class OfflineWenetCtcModelConfig {
    private final String model;

    private OfflineWenetCtcModelConfig(Builder builder) {
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

        public OfflineWenetCtcModelConfig build() {
            return new OfflineWenetCtcModelConfig(this);
        }

        public Builder setModel(String model) {
            this.model = model;
            return this;
        }
    }
}
