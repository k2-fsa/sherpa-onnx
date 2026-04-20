// Copyright 2026 Xiaomi Corporation
package com.k2fsa.sherpa.onnx;

public class OfflineTtsSupertonicModelConfig {
    private final String durationPredictor;
    private final String textEncoder;
    private final String vectorEstimator;
    private final String vocoder;
    private final String ttsJson;
    private final String unicodeIndexer;
    private final String voiceStyle;

    private OfflineTtsSupertonicModelConfig(Builder builder) {
        this.durationPredictor = builder.durationPredictor;
        this.textEncoder = builder.textEncoder;
        this.vectorEstimator = builder.vectorEstimator;
        this.vocoder = builder.vocoder;
        this.ttsJson = builder.ttsJson;
        this.unicodeIndexer = builder.unicodeIndexer;
        this.voiceStyle = builder.voiceStyle;
    }

    public String getDurationPredictor() {
        return durationPredictor;
    }

    public String getTextEncoder() {
        return textEncoder;
    }

    public String getVectorEstimator() {
        return vectorEstimator;
    }

    public String getVocoder() {
        return vocoder;
    }

    public String getTtsJson() {
        return ttsJson;
    }

    public String getUnicodeIndexer() {
        return unicodeIndexer;
    }

    public String getVoiceStyle() {
        return voiceStyle;
    }

    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private String durationPredictor = "";
        private String textEncoder = "";
        private String vectorEstimator = "";
        private String vocoder = "";
        private String ttsJson = "";
        private String unicodeIndexer = "";
        private String voiceStyle = "";

        public OfflineTtsSupertonicModelConfig build() {
            return new OfflineTtsSupertonicModelConfig(this);
        }

        public Builder setDurationPredictor(String durationPredictor) {
            this.durationPredictor = durationPredictor;
            return this;
        }

        public Builder setTextEncoder(String textEncoder) {
            this.textEncoder = textEncoder;
            return this;
        }

        public Builder setVectorEstimator(String vectorEstimator) {
            this.vectorEstimator = vectorEstimator;
            return this;
        }

        public Builder setVocoder(String vocoder) {
            this.vocoder = vocoder;
            return this;
        }

        public Builder setTtsJson(String ttsJson) {
            this.ttsJson = ttsJson;
            return this;
        }

        public Builder setUnicodeIndexer(String unicodeIndexer) {
            this.unicodeIndexer = unicodeIndexer;
            return this;
        }

        public Builder setVoiceStyle(String voiceStyle) {
            this.voiceStyle = voiceStyle;
            return this;
        }
    }
}
