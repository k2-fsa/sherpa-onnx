// Copyright 2025 Xiaomi Corporation

package com.k2fsa.sherpa.onnx;

public class OfflineTtsMatchaModelConfig {
    private final String acousticModel;
    private final String vocoder;
    private final String lexicon;
    private final String tokens;
    private final String dataDir;
    private final String dictDir;
    private final float noiseScale;
    private final float lengthScale;

    private OfflineTtsMatchaModelConfig(Builder builder) {
        this.acousticModel = builder.acousticModel;
        this.vocoder = builder.vocoder;
        this.lexicon = builder.lexicon;
        this.tokens = builder.tokens;
        this.dataDir = builder.dataDir;
        this.dictDir = builder.dictDir;
        this.noiseScale = builder.noiseScale;
        this.lengthScale = builder.lengthScale;
    }

    public static Builder builder() {
        return new Builder();
    }

    public String getAcousticModel() {
        return acousticModel;
    }

    public String getVocoder() {
        return vocoder;
    }

    public String getLexicon() {
        return lexicon;
    }

    public String getTokens() {
        return tokens;
    }

    public String getDataDir() {
        return dataDir;
    }

    public String getDictDir() {
        return dictDir;
    }

    public float getLengthScale() {
        return lengthScale;
    }

    public float getNoiseScale() {
        return noiseScale;
    }

    public static class Builder {
        private String acousticModel = "";
        private String vocoder = "";
        private String lexicon = "";
        private String tokens = "";
        private String dataDir = "";
        private String dictDir = "";
        private float noiseScale = 1.0f;
        private float lengthScale = 1.0f;

        public OfflineTtsMatchaModelConfig build() {
            return new OfflineTtsMatchaModelConfig(this);
        }

        public Builder setAcousticModel(String acousticModel) {
            this.acousticModel = acousticModel;
            return this;
        }

        public Builder setVocoder(String vocoder) {
            this.vocoder = vocoder;
            return this;
        }

        public Builder setTokens(String tokens) {
            this.tokens = tokens;
            return this;
        }

        public Builder setLexicon(String lexicon) {
            this.lexicon = lexicon;
            return this;
        }

        public Builder setDataDir(String dataDir) {
            this.dataDir = dataDir;
            return this;
        }

        public Builder setDictDir(String dictDir) {
            this.dictDir = dictDir;
            return this;
        }

        public Builder setNoiseScale(float noiseScale) {
            this.noiseScale = noiseScale;
            return this;
        }

        public Builder setLengthScale(float lengthScale) {
            this.lengthScale = lengthScale;
            return this;
        }
    }
}
