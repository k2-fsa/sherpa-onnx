/*
 * // Copyright 2022-2023 by zhaoming
 */

package com.k2fsa.sherpa.onnx;

public class OnlineRecognizerConfig {
    private final FeatureConfig featConfig;
    private final OnlineModelConfig modelConfig;
    private final EndpointConfig endpointConfig;
    private final OnlineLMConfig lmConfig;
    private final boolean enableEndpoint;
    private final String decodingMethod;
    private final int maxActivePaths;
    private final String hotwordsFile;
    private final float hotwordsScore;

    public OnlineRecognizerConfig(
            FeatureConfig featConfig,
            OnlineModelConfig modelConfig,
            EndpointConfig endpointConfig,
            OnlineLMConfig lmConfig,
            boolean enableEndpoint,
            String decodingMethod,
            int maxActivePaths,
            String hotwordsFile,
            float hotwordsScore) {
        this.featConfig = featConfig;
        this.modelConfig = modelConfig;
        this.endpointConfig = endpointConfig;
        this.lmConfig = lmConfig;
        this.enableEndpoint = enableEndpoint;
        this.decodingMethod = decodingMethod;
        this.maxActivePaths = maxActivePaths;
        this.hotwordsFile = hotwordsFile;
        this.hotwordsScore = hotwordsScore;
    }

    public OnlineLMConfig getLmConfig() {
        return lmConfig;
    }

    public FeatureConfig getFeatConfig() {
        return featConfig;
    }

    public OnlineModelConfig getModelConfig() {
        return modelConfig;
    }

    public EndpointConfig getEndpointConfig() {
        return endpointConfig;
    }

    public boolean isEnableEndpoint() {
        return enableEndpoint;
    }

    public String getDecodingMethod() {
        return decodingMethod;
    }

    public int getMaxActivePaths() {
        return maxActivePaths;
    }
}
