// Copyright 2025 Xiaomi Corporation

package com.k2fsa.sherpa.onnx;

public class OfflineDolphinModelConfig {
    private final String model;
    private final String encoder;
    private final String decoder;
    private final String language;
    private final String region;

    private OfflineDolphinModelConfig(Builder builder) {
        this.model = builder.model;
        this.encoder = builder.encoder;
        this.decoder = builder.decoder;
        this.language = builder.language;
        this.region = builder.region;
    }

    public static Builder builder() {
        return new Builder();
    }

    public String getModel() {
        return model;
    }

    public String getEncoder() {
        return encoder;
    }

    public String getDecoder() {
        return decoder;
    }

    public String getLanguage() {
        return language;
    }

    public String getRegion() {
        return region;
    }

    public static class Builder {
        private String model = "";
        private String encoder = "";
        private String decoder = "";
        private String language = "";
        private String region = "";

        public OfflineDolphinModelConfig build() {
            return new OfflineDolphinModelConfig(this);
        }

        public Builder setModel(String model) {
            this.model = model;
            return this;
        }

        public Builder setEncoder(String encoder) {
            this.encoder = encoder;
            return this;
        }

        public Builder setDecoder(String decoder) {
            this.decoder = decoder;
            return this;
        }

        public Builder setLanguage(String language) {
            this.language = language;
            return this;
        }

        public Builder setRegion(String region) {
            this.region = region;
            return this;
        }
    }
}
