/*
 * // Copyright 2022-2023 by zhaoming
 */

package com.k2fsa.sherpa.onnx;

public class OnlineRecognizerConfig {
  private final FeatureConfig featConfig;
  private final OnlineTransducerModelConfig modelConfig;
  private final EndpointConfig endpointConfig;
  private final OnlineLMConfig lmConfig;
  private final boolean enableEndpoint;
  private final String decodingMethod;
  private final int maxActivePaths;
  

  public OnlineRecognizerConfig(
      FeatureConfig featConfig,
      OnlineTransducerModelConfig modelConfig,
      EndpointConfig endpointConfig,
	  OnlineLMConfig lmConfig,
      boolean enableEndpoint,
      String decodingMethod,
      int maxActivePaths) {
    this.featConfig = featConfig;
    this.modelConfig = modelConfig;
    this.endpointConfig = endpointConfig;
	this.lmConfig = lmConfig;
    this.enableEndpoint = enableEndpoint;
    this.decodingMethod = decodingMethod;
    this.maxActivePaths = maxActivePaths;
  }

  public OnlineLMConfig getLmConfig() {
    return lmConfig;
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
