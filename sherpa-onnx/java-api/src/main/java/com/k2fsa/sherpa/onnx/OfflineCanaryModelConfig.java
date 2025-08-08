// Copyright 2025 Xiaomi Corporation

package com.k2fsa.sherpa.onnx;

public class OfflineCanaryModelConfig {
    private final String encoder;
    private final String decoder;
    private final String srcLang;
    private final String tgtLang;
    private final boolean usePnc;

    private OfflineCanaryModelConfig(Builder builder) {
        this.encoder = builder.encoder;
        this.decoder = builder.decoder;
        this.srcLang = builder.srcLang;
        this.tgtLang = builder.tgtLang;
        this.usePnc = builder.usePnc;
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

    public String getSrcLang() {
        return srcLang;
    }

    public String getTgtLang() {
        return tgtLang;
    }

    public boolean isUsePnc() {
        return usePnc;
    }

    public static class Builder {
        private String encoder = "";
        private String decoder = "";
        private String srcLang = "en";
        private String tgtLang = "en";
        private boolean usePnc = true;

        public OfflineCanaryModelConfig build() {
            return new OfflineCanaryModelConfig(this);
        }

        public Builder setEncoder(String encoder) {
            this.encoder = encoder;
            return this;
        }

        public Builder setDecoder(String decoder) {
            this.decoder = decoder;
            return this;
        }

        public Builder setSrcLang(String srcLang) {
            this.srcLang = srcLang;
            return this;
        }

        public Builder setTgtLang(String tgtLang) {
            this.tgtLang = tgtLang;
            return this;
        }

        public Builder setUsePnc(boolean usePnc) {
            this.usePnc = usePnc;
            return this;
        }
    }
}
