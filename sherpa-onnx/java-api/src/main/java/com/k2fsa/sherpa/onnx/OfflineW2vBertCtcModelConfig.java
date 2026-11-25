package com.k2fsa.sherpa.onnx;

public class OfflineW2vBertCtcModelConfig {
    private final String model;

    private OfflineW2vBertCtcModelConfig(Builder builder) {
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

        public OfflineW2vBertCtcModelConfig build() {
            return new OfflineW2vBertCtcModelConfig(this);
        }

        public Builder setModel(String model) {
            this.model = model;
            return this;
        }
    }
}
