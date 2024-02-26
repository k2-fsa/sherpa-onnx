/*
 * // Copyright 2022-2023 by zhaoming
 */

package com.k2fsa.sherpa.onnx;

public class OnlineLMConfig {
    private final String model;
    private final float scale;

    public OnlineLMConfig(String model, float scale) {
        this.model = model;
        this.scale = scale;
    }

    public String getModel() {
        return model;
    }

    public float getScale() {
        return scale;
    }
}
