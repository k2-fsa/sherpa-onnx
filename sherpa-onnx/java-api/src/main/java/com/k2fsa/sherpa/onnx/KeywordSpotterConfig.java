// Copyright 2024 Xiaomi Corporation

package com.k2fsa.sherpa.onnx;

public class KeywordSpotterConfig {
    private final FeatureConfig featConfig;
    private final OnlineModelConfig modelConfig;

    private final int maxActivePaths;
    private final String keywordsFile;
    private final float keywordsScore;
    private final float keywordsThreshold;
    private final int numTrailingBlanks;

    private KeywordSpotterConfig(Builder builder) {
        this.featConfig = builder.featConfig;
        this.modelConfig = builder.modelConfig;
        this.maxActivePaths = builder.maxActivePaths;
        this.keywordsFile = builder.keywordsFile;
        this.keywordsScore = builder.keywordsScore;
        this.keywordsThreshold = builder.keywordsThreshold;
        this.numTrailingBlanks = builder.numTrailingBlanks;
    }

    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private FeatureConfig featConfig = FeatureConfig.builder().build();
        private OnlineModelConfig modelConfig = OnlineModelConfig.builder().build();
        private int maxActivePaths = 4;
        private String keywordsFile = "keywords.txt";
        private float keywordsScore = 1.5f;
        private float keywordsThreshold = 0.25f;
        private int numTrailingBlanks = 2;

        public KeywordSpotterConfig build() {
            return new KeywordSpotterConfig(this);
        }

        public Builder setFeatureConfig(FeatureConfig featConfig) {
            this.featConfig = featConfig;
            return this;
        }

        public Builder setOnlineModelConfig(OnlineModelConfig modelConfig) {
            this.modelConfig = modelConfig;
            return this;
        }

        public Builder setMaxActivePaths(int maxActivePaths) {
            this.maxActivePaths = maxActivePaths;
            return this;
        }

        public Builder setKeywordsFile(String keywordsFile) {
            this.keywordsFile = keywordsFile;
            return this;
        }

        public Builder setKeywordsScore(float keywordsScore) {
            this.keywordsScore = keywordsScore;
            return this;
        }

        public Builder setKeywordsThreshold(float keywordsThreshold) {
            this.keywordsThreshold = keywordsThreshold;
            return this;
        }

        public Builder setNumTrailingBlanks(int numTrailingBlanks) {
            this.numTrailingBlanks = numTrailingBlanks;
            return this;
        }
    }
}
