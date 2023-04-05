/*
 * // Copyright 2022-2023 by zhaoming
 */

package com.k2fsa.sherpaonnx.rcglib;

public class OnlineRecognizerConfig {
    final private FeatureConfig featConfig;
    final private OnlineTransducerModelConfig modelConfig;
    final private EndpointConfig endpointConfig;
    final private boolean enableEndpoint;
    final private String decodingMethod;
    final private int maxActivePaths;

    public OnlineRecognizerConfig(FeatureConfig featConfig, OnlineTransducerModelConfig modelConfig, EndpointConfig endpointConfig, boolean enableEndpoint, String decodingMethod, int maxActivePaths) {
        this.featConfig = featConfig;
        this.modelConfig = modelConfig;
        this.endpointConfig = endpointConfig;
        this.enableEndpoint = enableEndpoint;
        this.decodingMethod = decodingMethod;
        this.maxActivePaths = maxActivePaths;
    }

    public FeatureConfig getFeatConfig() {
        return featConfig;
    }

    public OnlineTransducerModelConfig getModelConfig() {
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
