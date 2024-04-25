// Copyright 2024 Xiaomi Corporation

package com.k2fsa.sherpa.onnx;

public class OfflineTtsConfig {
    private final OfflineTtsModelConfig model;
    private final String ruleFsts;
    private final String ruleFars;
    private final int maxNumSentences;

    private OfflineTtsConfig(Builder builder) {
        this.model = builder.model;
        this.ruleFsts = builder.ruleFsts;
        this.ruleFars = builder.ruleFars;
        this.maxNumSentences = builder.maxNumSentences;
    }

    public static Builder builder() {
        return new Builder();
    }

    public OfflineTtsModelConfig getModel() {
        return model;
    }

    public String getRuleFsts() {
        return ruleFsts;
    }

    public String getRuleFars() {
        return ruleFars;
    }

    public int getMaxNumSentences() {
        return maxNumSentences;
    }

    public static class Builder {
        private OfflineTtsModelConfig model = OfflineTtsModelConfig.builder().build();
        private String ruleFsts = "";
        private String ruleFars = "";
        private int maxNumSentences = 1;

        public OfflineTtsConfig build() {
            return new OfflineTtsConfig(this);
        }

        public Builder setModel(OfflineTtsModelConfig model) {
            this.model = model;
            return this;
        }

        public Builder setRuleFsts(String ruleFsts) {
            this.ruleFsts = ruleFsts;
            return this;
        }

        public Builder setRuleFars(String ruleFars) {
            this.ruleFars = ruleFars;
            return this;
        }

        public Builder setMaxNumSentences(int maxNumSentences) {
            this.maxNumSentences = maxNumSentences;
            return this;
        }
    }
}
