package com.k2fsa.sherpa.onnx;

public class OfflineSpeakerDiarizationConfig {
    private final OfflineSpeakerSegmentationModelConfig segmentation;
    private final SpeakerEmbeddingExtractorConfig embedding;
    private final FastClusteringConfig clustering;
    private final float minDurationOn;
    private final float minDurationOff;

    private OfflineSpeakerDiarizationConfig(Builder builder) {
        this.segmentation = builder.segmentation;
        this.embedding = builder.embedding;
        this.clustering = builder.clustering;
        this.minDurationOff = builder.minDurationOff;
        this.minDurationOn = builder.minDurationOn;
    }

    public static Builder builder() {
        return new Builder();
    }

    public OfflineSpeakerSegmentationModelConfig getSegmentation() {
        return segmentation;
    }

    public SpeakerEmbeddingExtractorConfig getEmbedding() {
        return embedding;
    }

    public FastClusteringConfig getClustering() {
        return clustering;
    }

    public float getMinDurationOff() {
        return minDurationOff;
    }

    public float getMinDurationOn() {
        return minDurationOn;
    }

    public static class Builder {
        private OfflineSpeakerSegmentationModelConfig segmentation = OfflineSpeakerSegmentationModelConfig.builder().build();
        private SpeakerEmbeddingExtractorConfig embedding = SpeakerEmbeddingExtractorConfig.builder().build();
        private FastClusteringConfig clustering = FastClusteringConfig.builder().build();
        private float minDurationOn = 0.2f;
        private float minDurationOff = 0.5f;

        public OfflineSpeakerDiarizationConfig build() {
            return new OfflineSpeakerDiarizationConfig(this);
        }

        public Builder setSegmentation(OfflineSpeakerSegmentationModelConfig segmentation) {
            this.segmentation = segmentation;
            return this;
        }

        public Builder setEmbedding(SpeakerEmbeddingExtractorConfig embedding) {
            this.embedding = embedding;
            return this;
        }

        public Builder setClustering(FastClusteringConfig clustering) {
            this.clustering = clustering;
            return this;
        }

        public Builder setMinDurationOff(float minDurationOff) {
            this.minDurationOff = minDurationOff;
            return this;
        }

        public Builder setMinDurationOn(float minDurationOn) {
            this.minDurationOn = minDurationOn;
            return this;
        }
    }

}
