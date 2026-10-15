// Copyright 2026 Silvio Tomatis

package com.k2fsa.sherpa.onnx;

/**
 * Sortformer end-to-end diarization model, e.g., Nemotron-3-Diarization.
 * If the model is set, pyannote, embedding and clustering are ignored.
 */
public class OfflineSpeakerSegmentationSortformerModelConfig {
    private final String model;
    private final float threshold;

    private OfflineSpeakerSegmentationSortformerModelConfig(Builder builder) {
        this.model = builder.model;
        this.threshold = builder.threshold;
    }

    public static Builder builder() {
        return new Builder();
    }

    public String getModel() {
        return model;
    }

    public float getThreshold() {
        return threshold;
    }

    public static class Builder {
        private String model = "";
        private float threshold = 0.5f;

        public OfflineSpeakerSegmentationSortformerModelConfig build() {
            return new OfflineSpeakerSegmentationSortformerModelConfig(this);
        }

        public Builder setModel(String model) {
            this.model = model;
            return this;
        }

        public Builder setThreshold(float value) {
            this.threshold = value;
            return this;
        }
    }
}
