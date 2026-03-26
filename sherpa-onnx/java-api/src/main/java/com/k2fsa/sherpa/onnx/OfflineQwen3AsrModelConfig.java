package com.k2fsa.sherpa.onnx;

public class OfflineQwen3AsrModelConfig {
    private final String convFrontend;
    private final String encoder;
    private final String decoder;
    private final String tokenizer;
    private final int maxTotalLen;
    private final int maxNewTokens;
    private final float temperature;
    private final float topP;
    private final int seed;

    private OfflineQwen3AsrModelConfig(Builder builder) {
        this.convFrontend = builder.convFrontend;
        this.encoder = builder.encoder;
        this.decoder = builder.decoder;
        this.tokenizer = builder.tokenizer;
        this.maxTotalLen = builder.maxTotalLen;
        this.maxNewTokens = builder.maxNewTokens;
        this.temperature = builder.temperature;
        this.topP = builder.topP;
        this.seed = builder.seed;
    }

    public static Builder builder() {
        return new Builder();
    }

    public String getConvFrontend() {
        return convFrontend;
    }

    public String getEncoder() {
        return encoder;
    }

    public String getDecoder() {
        return decoder;
    }

    public String getTokenizer() {
        return tokenizer;
    }

    public int getMaxTotalLen() {
        return maxTotalLen;
    }

    public int getMaxNewTokens() {
        return maxNewTokens;
    }

    public float getTemperature() {
        return temperature;
    }

    public float getTopP() {
        return topP;
    }

    public int getSeed() {
        return seed;
    }

    public static class Builder {
        private String convFrontend = "";
        private String encoder = "";
        private String decoder = "";
        private String tokenizer = "";
        private int maxTotalLen = 512;
        private int maxNewTokens = 128;
        private float temperature = 1e-6f;
        private float topP = 0.8f;
        private int seed = 42;

        public OfflineQwen3AsrModelConfig build() {
            return new OfflineQwen3AsrModelConfig(this);
        }

        public Builder setConvFrontend(String convFrontend) {
            this.convFrontend = convFrontend;
            return this;
        }

        public Builder setEncoder(String encoder) {
            this.encoder = encoder;
            return this;
        }

        public Builder setDecoder(String decoder) {
            this.decoder = decoder;
            return this;
        }

        public Builder setTokenizer(String tokenizer) {
            this.tokenizer = tokenizer;
            return this;
        }

        public Builder setMaxTotalLen(int maxTotalLen) {
            this.maxTotalLen = maxTotalLen;
            return this;
        }

        public Builder setMaxNewTokens(int maxNewTokens) {
            this.maxNewTokens = maxNewTokens;
            return this;
        }

        public Builder setTemperature(float temperature) {
            this.temperature = temperature;
            return this;
        }

        public Builder setTopP(float topP) {
            this.topP = topP;
            return this;
        }

        public Builder setSeed(int seed) {
            this.seed = seed;
            return this;
        }
    }
}
