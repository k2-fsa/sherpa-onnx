package com.k2fsa.sherpa.onnx;

public class OnlineToneCtcModelConfig {
    private final String model;

    private OnlineToneCtcModelConfig(Builder builder) {
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

        public OnlineToneCtcModelConfig build() {
            return new OnlineToneCtcModelConfig(this);
        }

        public Builder setModel(String model) {
            this.model = model;
            return this;
        }
    }
}
