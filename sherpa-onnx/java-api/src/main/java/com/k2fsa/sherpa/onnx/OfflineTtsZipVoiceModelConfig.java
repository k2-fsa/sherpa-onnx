// Copyright 2026 Xiaomi Corporation

package com.k2fsa.sherpa.onnx;

public class OfflineTtsZipVoiceModelConfig {
    private final String tokens;
    private final String encoder;
    private final String decoder;
    private final String vocoder;
    private final String dataDir;
    private final String lexicon;
    private final float featScale;
    private final float tShift;
    private final float targetRms;
    private final float guidanceScale;

    private OfflineTtsZipVoiceModelConfig(Builder builder) {
        this.tokens = builder.tokens;
        this.encoder = builder.encoder;
        this.decoder = builder.decoder;
        this.vocoder = builder.vocoder;
        this.dataDir = builder.dataDir;
        this.lexicon = builder.lexicon;
        this.featScale = builder.featScale;
        this.tShift = builder.tShift;
        this.targetRms = builder.targetRms;
        this.guidanceScale = builder.guidanceScale;
    }

    public static Builder builder() {
        return new Builder();
    }

    public String getTokens() {
        return tokens;
    }

    public String getEncoder() {
        return encoder;
    }

    public String getDecoder() {
        return decoder;
    }

    public String getVocoder() {
        return vocoder;
    }

    public String getDataDir() {
        return dataDir;
    }

    public String getLexicon() {
        return lexicon;
    }

    public float getFeatScale() {
        return featScale;
    }

    public float getTShift() {
        return tShift;
    }

    public float getTargetRms() {
        return targetRms;
    }

    public float getGuidanceScale() {
        return guidanceScale;
    }

    public static class Builder {
        private String tokens = "";
        private String encoder = "";
        private String decoder = "";
        private String vocoder = "";
        private String dataDir = "";
        private String lexicon = "";
        private float featScale = 0.1f;
        private float tShift = 0.5f;
        private float targetRms = 0.1f;
        private float guidanceScale = 1.0f;

        public OfflineTtsZipVoiceModelConfig build() {
            return new OfflineTtsZipVoiceModelConfig(this);
        }

        public Builder setTokens(String tokens) {
            this.tokens = tokens;
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

        public Builder setVocoder(String vocoder) {
            this.vocoder = vocoder;
            return this;
        }

        public Builder setDataDir(String dataDir) {
            this.dataDir = dataDir;
            return this;
        }

        public Builder setLexicon(String lexicon) {
            this.lexicon = lexicon;
            return this;
        }

        public Builder setFeatScale(float featScale) {
            this.featScale = featScale;
            return this;
        }

        public Builder setTShift(float tShift) {
            this.tShift = tShift;
            return this;
        }

        public Builder setTargetRms(float targetRms) {
            this.targetRms = targetRms;
            return this;
        }

        public Builder setGuidanceScale(float guidanceScale) {
            this.guidanceScale = guidanceScale;
            return this;
        }
    }
}
