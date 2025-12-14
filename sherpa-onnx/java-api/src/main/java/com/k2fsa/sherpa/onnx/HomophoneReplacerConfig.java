// Copyright 2025 Xiaomi Corporation

package com.k2fsa.sherpa.onnx;

public class HomophoneReplacerConfig {
    private final String dictDir;  // unused
    private final String lexicon;
    private final String ruleFsts;

    private HomophoneReplacerConfig(Builder builder) {
        this.dictDir = builder.dictDir;
        this.lexicon = builder.lexicon;
        this.ruleFsts = builder.ruleFsts;
    }

    public static Builder builder() {
        return new Builder();
    }

    public String getDictDir() {
        return dictDir;
    }

    public String getLexicon() {
        return lexicon;
    }

    public String getRuleFsts() {
        return ruleFsts;
    }

    public static class Builder {
        private String dictDir = "";
        private String lexicon = "";
        private String ruleFsts = "";

        public HomophoneReplacerConfig build() {
            return new HomophoneReplacerConfig(this);
        }

        public Builder setDictDir(String dictDir) {
            this.dictDir = dictDir;
            return this;
        }

        public Builder setLexicon(String lexicon) {
            this.lexicon = lexicon;
            return this;
        }

        public Builder setRuleFsts(String ruleFsts) {
            this.ruleFsts = ruleFsts;
            return this;
        }
    }
}
