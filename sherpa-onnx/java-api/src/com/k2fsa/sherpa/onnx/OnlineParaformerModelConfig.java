// Copyright 2022-2023 by zhaoming
// Copyright 2024 Xiaomi Corporation

package com.k2fsa.sherpa.onnx;

public class OnlineParaformerModelConfig {
    private final String encoder;
    private final String decoder;

    private OnlineParaformerModelConfig(Builder builder) {
        this.encoder = builder.encoder;
        this.decoder = builder.decoder;
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

    public static class Builder {
        private String encoder = "";
        private String decoder = "";

        public OnlineParaformerModelConfig build() {
            return new OnlineParaformerModelConfig(this);
        }

        public Builder setEncoder(String encoder) {
            this.encoder = encoder;
            return this;
        }

        public Builder setDecoder(String decoder) {
            this.decoder = decoder;
            return this;
        }
    }
}
