// Copyright 2024 Xiaomi Corporation

package com.k2fsa.sherpa.onnx;

public class OfflineTransducerModelConfig {
    private final String encoder;
    private final String decoder;
    private final String joiner;
    private final QnnConfig qnnConfig;

    private OfflineTransducerModelConfig(Builder builder) {
        this.encoder = builder.encoder;
        this.decoder = builder.decoder;
        this.joiner = builder.joiner;
        this.qnnConfig = builder.qnnConfig;
    }

    public static Builder builder() {
        return new Builder();
    }

    public String getEncoder() {
        return encoder;
    }

    public String getDecoder() {
        return decoder;
    }

    public String getJoiner() {
        return joiner;
    }

    public QnnConfig getQnnConfig() {
        return qnnConfig;
    }

    public static class Builder {
        private String encoder = "";
        private String decoder = "";
        private String joiner = "";
        private QnnConfig qnnConfig = QnnConfig.builder().build();

        public OfflineTransducerModelConfig build() {
            return new OfflineTransducerModelConfig(this);
        }

        public Builder setEncoder(String encoder) {
            this.encoder = encoder;
            return this;
        }

        public Builder setDecoder(String decoder) {
            this.decoder = decoder;
            return this;
        }

        public Builder setJoiner(String joiner) {
            this.joiner = joiner;
            return this;
        }

        public Builder setQnnConfig(QnnConfig qnnConfig) {
            this.qnnConfig = qnnConfig;
            return this;
        }
    }
}
