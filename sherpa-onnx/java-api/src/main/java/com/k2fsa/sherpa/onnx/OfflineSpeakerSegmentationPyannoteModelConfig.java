// Copyright 2024 Xiaomi Corporation

package com.k2fsa.sherpa.onnx;

public class OfflineSpeakerSegmentationPyannoteModelConfig {
    private final String model;
    private final float windowShiftRatio;

    private OfflineSpeakerSegmentationPyannoteModelConfig(Builder builder) {
        this.model = builder.model;
        this.windowShiftRatio = builder.windowShiftRatio;
    }

    public static Builder builder() {
        return new Builder();
    }

    public String getModel() {
        return model;
    }

    public float getWindowShiftRatio() {
        return windowShiftRatio;
    }

    public static class Builder {
        private String model = "";
        private float windowShiftRatio = 0.1f;

        public OfflineSpeakerSegmentationPyannoteModelConfig build() {
            return new OfflineSpeakerSegmentationPyannoteModelConfig(this);
        }

        public Builder setModel(String model) {
            this.model = model;
            return this;
        }

        public Builder setWindowShiftRatio(float value) {
            this.windowShiftRatio = value;
            return this;
        }
    }
}
