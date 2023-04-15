/*
 * // Copyright 2022-2023 by zhaoming
 */

package com.k2fsa.sherpa.onnx;

public class FeatureConfig {
  private final int sampleRate;
  private final int featureDim;

  public FeatureConfig(int sampleRate, int featureDim) {
    this.sampleRate = sampleRate;
    this.featureDim = featureDim;
  }

  public int getSampleRate() {
    return sampleRate;
  }

  public int getFeatureDim() {
    return featureDim;
  }
}
