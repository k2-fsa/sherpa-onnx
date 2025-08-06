// Copyright 2022-2023 by zhaoming
// Copyright 2024 Xiaomi Corporation

package com.k2fsa.sherpa.onnx;

public class OnlineParaformerModelConfig {
    private final String encoder;
    private final String decoder;

    private OnlineParaformerModelConfig(Builder builder) {
      this.encoder = java.util.Objects.requireNonNull(builder.encoder, "encoder cannot be null");
      this.decoder = java.util.Objects.requireNonNull(builder.decoder, "decoder cannot be null");

      if (this.encoder.isEmpty() || this.decoder.isEmpty()) {
          throw new IllegalArgumentException("encoder/decoder path must not be empty");
      }
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
