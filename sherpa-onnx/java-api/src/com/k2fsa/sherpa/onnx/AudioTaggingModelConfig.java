// Copyright 2024 Xiaomi Corporation

package com.k2fsa.sherpa.onnx;

public class AudioTaggingModelConfig {
    private final OfflineZipformerAudioTaggingModelConfig zipformer;
    private final String ced;
    private final int numThreads;
    private final boolean debug;
    private final String provider;

    private AudioTaggingModelConfig(Builder builder) {
        this.zipformer = builder.zipformer;
        this.ced = builder.ced;
        this.numThreads = builder.numThreads;
        this.debug = builder.debug;
        this.provider = builder.provider;
    }

    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private OfflineZipformerAudioTaggingModelConfig zipformer = OfflineZipformerAudioTaggingModelConfig.builder().build();
        private String ced = "";
        private int numThreads = 1;
        private boolean debug = true;
        private String provider = "cpu";

        public AudioTaggingModelConfig build() {
            return new AudioTaggingModelConfig(this);
        }

        public Builder setZipformer(OfflineZipformerAudioTaggingModelConfig zipformer) {
            this.zipformer = zipformer;
            return this;
        }

        public Builder setCED(String ced) {
            this.ced = ced;
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
