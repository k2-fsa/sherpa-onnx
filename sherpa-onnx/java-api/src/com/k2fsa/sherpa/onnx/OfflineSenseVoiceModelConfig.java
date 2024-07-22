// Copyright 2024 Xiaomi Corporation

package com.k2fsa.sherpa.onnx;

public class OfflineSenseVoiceModelConfig {
    private final String model;
    private final String language;
    private final boolean useInverseTextNormalization;

    private OfflineSenseVoiceModelConfig(Builder builder) {
        this.model = builder.model;
        this.language = builder.language;
        this.useInverseTextNormalization = builder.useInverseTextNormalization;
    }

    public static Builder builder() {
        return new Builder();
    }

    public String getModel() {
        return model;
    }

    public String getLanguage() {
        return language;
    }

    public boolean getUseInverseTextNormalization() {
        return useInverseTextNormalization;
    }

    public static class Builder {
        private String model = "";
        private String language = "";
        private boolean useInverseTextNormalization = true;

        public OfflineSenseVoiceModelConfig build() {
            return new OfflineSenseVoiceModelConfig(this);
        }

        public Builder setModel(String model) {
            this.model = model;
            return this;
        }

        public Builder setLanguage(String language) {
            this.language = language;
            return this;
        }

        public Builder setInverseTextNormalization(boolean useInverseTextNormalization) {
            this.useInverseTextNormalization = useInverseTextNormalization;
            return this;
        }
    }
}
