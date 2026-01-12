package com.k2fsa.sherpa.onnx;

public class OfflineFunAsrNanoModelConfig {
    private final String encoderAdaptor;
    private final String llm;
    private final String embedding;
    private final String tokenizer;
    private final String systemPrompt;
    private final String userPrompt;
    private final int maxNewTokens;
    private final float temperature;
    private final float topP;
    private final int seed;

    private OfflineFunAsrNanoModelConfig(Builder builder) {
        this.encoderAdaptor = builder.encoderAdaptor;
        this.llm = builder.llm;
        this.embedding = builder.embedding;
        this.tokenizer = builder.tokenizer;
        this.systemPrompt = builder.systemPrompt;
        this.userPrompt = builder.userPrompt;
        this.maxNewTokens = builder.maxNewTokens;
        this.temperature = builder.temperature;
        this.topP = builder.topP;
        this.seed = builder.seed;
    }

    public static Builder builder() {
        return new Builder();
    }

    public String getEncoderAdaptor() {
        return encoderAdaptor;
    }

    public String getLLM() {
        return llm;
    }

    public String getEmbedding() {
        return embedding;
    }

    public String getTokenizer() {
        return tokenizer;
    }

    public String getSystemPrompt() {
        return systemPrompt;
    }

    public String getUserPrompt() {
        return userPrompt;
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
        private String encoderAdaptor = "";
        private String llm = "";
        private String embedding = "";
        private String tokenizer = "";
        private String systemPrompt = "You are a helpful assistant.";
        private String userPrompt = "语音转写：";
        private int maxNewTokens = 512;
        private float temperature = 1e-6f;
        private float topP = 0.8f;
        private int seed = 42;

        public OfflineFunAsrNanoModelConfig build() {
            return new OfflineFunAsrNanoModelConfig(this);
        }

        public Builder setEncoderAdaptor(String encoderAdaptor) {
            this.encoderAdaptor = encoderAdaptor;
            return this;
        }

        public Builder setLLM(String llm) {
            this.llm = llm;
            return this;
        }

        public Builder setEmbedding(String embedding) {
            this.embedding = embedding;
            return this;
        }

        public Builder setTokenizer(String tokenizer) {
            this.tokenizer = tokenizer;
            return this;
        }

        public Builder setSystemPrompt(String systemPrompt) {
            this.systemPrompt = systemPrompt;
            return this;
        }

        public Builder setUserPrompt(String userPrompt) {
            this.userPrompt = userPrompt;
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
