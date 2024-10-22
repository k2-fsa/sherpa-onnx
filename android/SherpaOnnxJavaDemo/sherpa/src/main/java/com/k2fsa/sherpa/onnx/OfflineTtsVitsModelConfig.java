// Copyright 2024 Xiaomi Corporation

package com.k2fsa.sherpa.onnx;

public class OfflineTtsVitsModelConfig {
    private final String model;
    private final String lexicon;
    private final String tokens;
    private final String dataDir;
    private final String dictDir;
    private final float noiseScale;
    private final float noiseScaleW;
    private final float lengthScale;

    private OfflineTtsVitsModelConfig(Builder builder) {
        this.model = builder.model;
        this.lexicon = builder.lexicon;
        this.tokens = builder.tokens;
        this.dataDir = builder.dataDir;
        this.dictDir = builder.dictDir;
        this.noiseScale = builder.noiseScale;
        this.noiseScaleW = builder.noiseScaleW;
        this.lengthScale = builder.lengthScale;
    }

    public static Builder builder() {
        return new Builder();
    }

    public String getModel() {
        return model;
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

    public float getNoiseScaleW() {
        return noiseScaleW;
    }

    public static class Builder {
        private String model;
        private String lexicon = "";
        private String tokens;
        private String dataDir = "";
        private String dictDir = "";
        private float noiseScale = 0.667f;
        private float noiseScaleW = 0.8f;
        private float lengthScale = 1.0f;

        public OfflineTtsVitsModelConfig build() {
            return new OfflineTtsVitsModelConfig(this);
        }

        public Builder setModel(String model) {
            this.model = model;
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

        public Builder setNoiseScaleW(float noiseScaleW) {
            this.noiseScaleW = noiseScaleW;
            return this;
        }

        public Builder setLengthScale(float lengthScale) {
            this.lengthScale = lengthScale;
            return this;
        }
    }
}
