// Copyright 2025 Xiaomi Corporation
package com.k2fsa.sherpa.onnx;

public class OfflineTtsKokoroModelConfig {
    private final String model;
    private final String voices;
    private final String tokens;
    private final String lexicon;
    private final String lang;
    private final String dataDir;
    private final String dictDir;  // unused
    private final float lengthScale;

    private OfflineTtsKokoroModelConfig(Builder builder) {
        this.model = builder.model;
        this.voices = builder.voices;
        this.tokens = builder.tokens;
        this.lexicon = builder.lexicon;
        this.lang = builder.lang;
        this.dataDir = builder.dataDir;
        this.dictDir = builder.dictDir;
        this.lengthScale = builder.lengthScale;
    }

    public static Builder builder() {
        return new Builder();
    }

    public String getModel() {
        return model;
    }

    public String getVoices() {
        return voices;
    }

    public String getTokens() {
        return tokens;
    }

    public String getDataDir() {
        return dataDir;
    }

    public float getLengthScale() {
        return lengthScale;
    }


    public static class Builder {
        private String model = "";
        private String voices = "";
        private String tokens = "";
        private String lexicon = "";
        private String lang = "";
        private String dataDir = "";
        private String dictDir = "";
        private float lengthScale = 1.0f;

        public OfflineTtsKokoroModelConfig build() {
            return new OfflineTtsKokoroModelConfig(this);
        }

        public Builder setModel(String model) {
            this.model = model;
            return this;
        }

        public Builder setVoices(String voices) {
            this.voices = voices;
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

        public Builder setLang(String lang) {
            this.lang = lang;
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

        public Builder setLengthScale(float lengthScale) {
            this.lengthScale = lengthScale;
            return this;
        }
    }
}
