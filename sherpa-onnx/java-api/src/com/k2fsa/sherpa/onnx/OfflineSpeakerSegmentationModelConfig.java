// Copyright 2024 Xiaomi Corporation

package com.k2fsa.sherpa.onnx;

public class OfflineSpeakerSegmentationModelConfig {
    private final OfflineSpeakerSegmentationPyannoteModelConfig pyannote;
    private final int numThreads;
    private final boolean debug;
    private final String provider;

    private OfflineSpeakerSegmentationModelConfig(Builder builder) {
        this.pyannote = builder.pyannote;
        this.numThreads = builder.numThreads;
        this.debug = builder.debug;
        this.provider = builder.provider;
    }

    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private OfflineSpeakerSegmentationPyannoteModelConfig pyannote = OfflineSpeakerSegmentationPyannoteModelConfig.builder().build();
        private int numThreads = 1;
        private boolean debug = true;
        private String provider = "cpu";

        public OfflineSpeakerSegmentationModelConfig build() {
            return new OfflineSpeakerSegmentationModelConfig(this);
        }

        public Builder setPyannote(OfflineSpeakerSegmentationPyannoteModelConfig pyannote) {
            this.pyannote = pyannote;
            return this;
        }

        public Builder setNumThreads(int numThreads) {
            this.numThreads = numThreads;
            return this;
        }

        public Builder setDebug(boolean debug) {
            this.debug = debug;
            return this;
        }

        public Builder setProvider(String provider) {
            this.provider = provider;
            return this;
        }
    }
}