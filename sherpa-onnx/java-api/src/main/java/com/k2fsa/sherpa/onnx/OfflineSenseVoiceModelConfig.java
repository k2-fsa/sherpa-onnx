// Copyright 2024 Xiaomi Corporation

package com.k2fsa.sherpa.onnx;

public class OfflineSenseVoiceModelConfig {
    private final String model;
    private final String language;
    private final boolean useInverseTextNormalization;
    private final QnnConfig qnnConfig;

    private OfflineSenseVoiceModelConfig(Builder builder) {
        this.model = builder.model;
        this.language = builder.language;
        this.useInverseTextNormalization = builder.useInverseTextNormalization;
        this.qnnConfig = builder.qnnConfig;
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

    public QnnConfig getQnnConfig() {
        return qnnConfig;
    }

    public static class Builder {
        private String model = "";
        private String language = "";
        private boolean useInverseTextNormalization = true;
        private QnnConfig qnnConfig = QnnConfig.builder().build();

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

        public Builder setQnnConfig(QnnConfig qnnConfig) {
            this.qnnConfig = qnnConfig;
            return this;
        }
    }
}
