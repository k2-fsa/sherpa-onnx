/*
 * // Copyright 2022-2023 by zhaoming
 */

package com.k2fsa.sherpaonnx;

public class FeatureConfig {
    final private int sampleRate;
    final private int featureDim;

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
