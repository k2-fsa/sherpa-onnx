// Copyright 2022-2023 by zhaoming
// Copyright 2024 Xiaomi Corporation

package com.k2fsa.sherpa.onnx;

public class OnlineRecognizerConfig {
    private final FeatureConfig featConfig;
    private final OnlineModelConfig modelConfig;
    private final OnlineLMConfig lmConfig;

    private final OnlineCtcFstDecoderConfig ctcFstDecoderConfig;
    private final EndpointConfig endpointConfig;
    private final boolean enableEndpoint;
    private final String decodingMethod;
    private final int maxActivePaths;
    private final String hotwordsFile;
    private final float hotwordsScore;
    private final String ruleFsts;
    private final String ruleFars;
    private final float blankPenalty;

    private OnlineRecognizerConfig(Builder builder) {
        this.featConfig = builder.featConfig;
        this.modelConfig = builder.modelConfig;
        this.lmConfig = builder.lmConfig;
        this.ctcFstDecoderConfig = builder.ctcFstDecoderConfig;
        this.endpointConfig = builder.endpointConfig;
        this.enableEndpoint = builder.enableEndpoint;
        this.decodingMethod = builder.decodingMethod;
        this.maxActivePaths = builder.maxActivePaths;
        this.hotwordsFile = builder.hotwordsFile;
        this.hotwordsScore = builder.hotwordsScore;
        this.ruleFsts = builder.ruleFsts;
        this.ruleFars = builder.ruleFars;
        this.blankPenalty = builder.blankPenalty;
    }

    public static Builder builder() {
        return new Builder();
    }

    public OnlineModelConfig getModelConfig() {
        return modelConfig;
    }

    public static class Builder {
        private FeatureConfig featConfig = FeatureConfig.builder().build();
        private OnlineModelConfig modelConfig = OnlineModelConfig.builder().build();
        private OnlineLMConfig lmConfig = OnlineLMConfig.builder().build();
        private OnlineCtcFstDecoderConfig ctcFstDecoderConfig = OnlineCtcFstDecoderConfig.builder().build();
        private EndpointConfig endpointConfig = EndpointConfig.builder().build();
        private boolean enableEndpoint = true;
        private String decodingMethod = "greedy_search";
        private int maxActivePaths = 4;
        private String hotwordsFile = "";
        private float hotwordsScore = 1.5f;
        private String ruleFsts = "";
        private String ruleFars = "";
        private float blankPenalty = 0.0f;

        public OnlineRecognizerConfig build() {
            return new OnlineRecognizerConfig(this);
        }

        public Builder setFeatureConfig(FeatureConfig featConfig) {
            this.featConfig = featConfig;
            return this;
        }

        public Builder setOnlineModelConfig(OnlineModelConfig modelConfig) {
            this.modelConfig = modelConfig;
            return this;
        }

        public Builder setOnlineLMConfig(OnlineLMConfig lmConfig) {
            this.lmConfig = lmConfig;
            return this;
        }

        public Builder setCtcFstDecoderConfig(OnlineCtcFstDecoderConfig ctcFstDecoderConfig) {
            this.ctcFstDecoderConfig = ctcFstDecoderConfig;
            return this;
        }

        public Builder setEndpointConfig(EndpointConfig endpointConfig) {
            this.endpointConfig = endpointConfig;
            return this;
        }

        public Builder setEnableEndpoint(boolean enableEndpoint) {
            this.enableEndpoint = enableEndpoint;
            return this;
        }

        public Builder setDecodingMethod(String decodingMethod) {
            this.decodingMethod = decodingMethod;
            return this;
        }

        public Builder setMaxActivePaths(int maxActivePaths) {
            this.maxActivePaths = maxActivePaths;
            return this;
        }

        public Builder setHotwordsFile(String hotwordsFile) {
            this.hotwordsFile = hotwordsFile;
            return this;
        }

        public Builder setHotwordsScore(float hotwordsScore) {
            this.hotwordsScore = hotwordsScore;
            return this;
        }

        public Builder setRuleFsts(String ruleFsts) {
            this.ruleFsts = ruleFsts;
            return this;
        }

        public Builder setRuleFars(String ruleFars) {
            this.ruleFars = ruleFars;
            return this;
        }

        public Builder setBlankPenalty(float blankPenalty) {
            this.blankPenalty = blankPenalty;
            return this;
        }
    }
}
