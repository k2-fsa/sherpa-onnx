// Copyright 2024 Xiaomi Corporation

package com.k2fsa.sherpa.onnx;

public class OfflineTtsModelConfig {
    private final OfflineTtsVitsModelConfig vits;
    private final OfflineTtsMatchaModelConfig matcha;
    private final OfflineTtsKokoroModelConfig kokoro;
    private final OfflineTtsKittenModelConfig kitten;
    private final OfflineTtsZipvoiceModelConfig zipvoice;
    private final int numThreads;
    private final boolean debug;
    private final String provider;

    private OfflineTtsModelConfig(Builder builder) {
        this.vits = builder.vits;
        this.matcha = builder.matcha;
        this.kokoro = builder.kokoro;
        this.kitten = builder.kitten;
        this.zipvoice = builder.zipvoice;
        this.numThreads = builder.numThreads;
        this.debug = builder.debug;
        this.provider = builder.provider;
    }

    public static Builder builder() {
        return new Builder();
    }

    public OfflineTtsVitsModelConfig getVits() {
        return vits;
    }

    public OfflineTtsMatchaModelConfig getMatcha() {
        return matcha;
    }

    public OfflineTtsKokoroModelConfig getKokoro() {
        return kokoro;
    }

    public OfflineTtsKittenModelConfig getKitten() {
        return kitten;
    }

    public OfflineTtsZipvoiceModelConfig getZipvoice() {
        return zipvoice;
    }

    public static class Builder {
        private OfflineTtsVitsModelConfig vits = OfflineTtsVitsModelConfig.builder().build();
        private OfflineTtsMatchaModelConfig matcha = OfflineTtsMatchaModelConfig.builder().build();
        private OfflineTtsKokoroModelConfig kokoro = OfflineTtsKokoroModelConfig.builder().build();
        private OfflineTtsKittenModelConfig kitten = OfflineTtsKittenModelConfig.builder().build();
        private OfflineTtsZipvoiceModelConfig zipvoice = OfflineTtsZipvoiceModelConfig.builder().build();
        private int numThreads = 1;
        private boolean debug = true;
        private String provider = "cpu";

        public OfflineTtsModelConfig build() {
            return new OfflineTtsModelConfig(this);
        }

        public Builder setVits(OfflineTtsVitsModelConfig vits) {
            this.vits = vits;
            return this;
        }

        public Builder setMatcha(OfflineTtsMatchaModelConfig matcha) {
            this.matcha = matcha;
            return this;
        }

        public Builder setKokoro(OfflineTtsKokoroModelConfig kokoro) {
            this.kokoro = kokoro;
            return this;
        }

        public Builder setKitten(OfflineTtsKittenModelConfig kitten) {
            this.kitten = kitten;
            return this;
        }

        public Builder setZipvoice(OfflineTtsZipvoiceModelConfig zipvoice) {
            this.zipvoice = zipvoice;
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
