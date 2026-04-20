// Copyright 2026 Xiaomi Corporation
package com.k2fsa.sherpa.onnx;

public class OfflineTtsPocketModelConfig {
    private final String lmFlow;
    private final String lmMain;
    private final String encoder;
    private final String decoder;
    private final String textConditioner;
    private final String vocabJson;
    private final String tokenScoresJson;
    private final int voiceEmbeddingCacheCapacity;

    private OfflineTtsPocketModelConfig(Builder builder) {
        this.lmFlow = builder.lmFlow;
        this.lmMain = builder.lmMain;
        this.encoder = builder.encoder;
        this.decoder = builder.decoder;
        this.textConditioner = builder.textConditioner;
        this.vocabJson = builder.vocabJson;
        this.tokenScoresJson = builder.tokenScoresJson;
        this.voiceEmbeddingCacheCapacity = builder.voiceEmbeddingCacheCapacity;
    }

    public String getLmFlow() {
        return lmFlow;
    }

    public String getLmMain() {
        return lmMain;
    }

    public String getEncoder() {
        return encoder;
    }

    public String getDecoder() {
        return decoder;
    }

    public String getTextConditioner() {
        return textConditioner;
    }

    public String getVocabJson() {
        return vocabJson;
    }

    public String getTokenScoresJson() {
        return tokenScoresJson;
    }

    public int getVoiceEmbeddingCacheCapacity() {
        return voiceEmbeddingCacheCapacity;
    }

    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private String lmFlow = "";
        private String lmMain = "";
        private String encoder = "";
        private String decoder = "";
        private String textConditioner = "";
        private String vocabJson = "";
        private String tokenScoresJson = "";
        private int voiceEmbeddingCacheCapacity = 50;

        public OfflineTtsPocketModelConfig build() {
            return new OfflineTtsPocketModelConfig(this);
        }

        public Builder setLmFlow(String lmFlow) {
            this.lmFlow = lmFlow;
            return this;
        }

        public Builder setLmMain(String lmMain) {
            this.lmMain = lmMain;
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

        public Builder setTextConditioner(String textConditioner) {
            this.textConditioner = textConditioner;
            return this;
        }

        public Builder setVocabJson(String vocabJson) {
            this.vocabJson = vocabJson;
            return this;
        }

        public Builder setTokenScoresJson(String tokenScoresJson) {
            this.tokenScoresJson = tokenScoresJson;
            return this;
        }

        public Builder setVoiceEmbeddingCacheCapacity(int voiceEmbeddingCacheCapacity) {
            this.voiceEmbeddingCacheCapacity = voiceEmbeddingCacheCapacity;
            return this;
        }
    }
}
