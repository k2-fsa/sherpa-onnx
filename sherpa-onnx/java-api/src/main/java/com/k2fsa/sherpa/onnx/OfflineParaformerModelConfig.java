// Copyright 2024 Xiaomi Corporation

package com.k2fsa.sherpa.onnx;

public class OfflineParaformerModelConfig {
    private final String model;
    private final QnnConfig qnnConfig;

    private OfflineParaformerModelConfig(Builder builder) {
        this.model = builder.model;
        this.qnnConfig = builder.qnnConfig;
    }

    public static Builder builder() {
        return new Builder();
    }

    public String getModel() {
        return model;
    }

    public QnnConfig getQnnConfig() {
        return qnnConfig;
    }

    public static class Builder {
        private String model = "";
        private QnnConfig qnnConfig = QnnConfig.builder().build();

        public OfflineParaformerModelConfig build() {
            return new OfflineParaformerModelConfig(this);
        }

        public Builder setModel(String model) {
            this.model = model;
            return this;
        }

        public Builder setQnnConfig(QnnConfig qnnConfig) {
            this.qnnConfig = qnnConfig;
            return this;
        }
    }
}
