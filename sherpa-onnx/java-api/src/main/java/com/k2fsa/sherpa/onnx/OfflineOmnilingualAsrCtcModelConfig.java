package com.k2fsa.sherpa.onnx;

public class OfflineOmnilingualAsrCtcModelConfig {
    private final String model;

    private OfflineOmnilingualAsrCtcModelConfig(Builder builder) {
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

        public OfflineOmnilingualAsrCtcModelConfig build() {
            return new OfflineOmnilingualAsrCtcModelConfig(this);
        }

        public Builder setModel(String model) {
            this.model = model;
            return this;
        }
    }
}
