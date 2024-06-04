// Copyright 2024 Xiaomi Corporation

package com.k2fsa.sherpa.onnx;

public class OfflineModelConfig {
    private final OfflineTransducerModelConfig transducer;
    private final OfflineParaformerModelConfig paraformer;
    private final OfflineWhisperModelConfig whisper;
    private final OfflineNemoEncDecCtcModelConfig nemo;
    private final String teleSpeech;
    private final String tokens;
    private final int numThreads;
    private final boolean debug;
    private final String provider;

    private final String modelType;
    private final String modelingUnit;
    private final String bpeVocab;

    private OfflineModelConfig(Builder builder) {
        this.transducer = builder.transducer;
        this.paraformer = builder.paraformer;
        this.whisper = builder.whisper;
        this.nemo = builder.nemo;
        this.teleSpeech = builder.teleSpeech;
        this.tokens = builder.tokens;
        this.numThreads = builder.numThreads;
        this.debug = builder.debug;
        this.provider = builder.provider;
        this.modelType = builder.modelType;
        this.modelingUnit = builder.modelingUnit;
        this.bpeVocab = builder.bpeVocab;
    }

    public static Builder builder() {
        return new Builder();
    }

    public OfflineParaformerModelConfig getParaformer() {
        return paraformer;
    }

    public OfflineTransducerModelConfig getTransducer() {
        return transducer;
    }

    public OfflineWhisperModelConfig getZipformer2Ctc() {
        return whisper;
    }

    public String getTokens() {
        return tokens;
    }

    public int getNumThreads() {
        return numThreads;
    }

    public boolean getDebug() {
        return debug;
    }

    public String getProvider() {
        return provider;
    }

    public String getModelType() {
        return modelType;
    }

    public String getModelingUnit() {
        return modelingUnit;
    }

    public String getBpeVocab() {
        return bpeVocab;
    }

    public String getTeleSpeech() {
        return teleSpeech;
    }

    public static class Builder {
        private OfflineParaformerModelConfig paraformer = OfflineParaformerModelConfig.builder().build();
        private OfflineTransducerModelConfig transducer = OfflineTransducerModelConfig.builder().build();
        private OfflineWhisperModelConfig whisper = OfflineWhisperModelConfig.builder().build();
        private OfflineNemoEncDecCtcModelConfig nemo = OfflineNemoEncDecCtcModelConfig.builder().build();
        private String teleSpeech = "";
        private String tokens = "";
        private int numThreads = 1;
        private boolean debug = true;
        private String provider = "cpu";
        private String modelType = "";
        private String modelingUnit = "cjkchar";
        private String bpeVocab = "";

        public OfflineModelConfig build() {
            return new OfflineModelConfig(this);
        }

        public Builder setTransducer(OfflineTransducerModelConfig transducer) {
            this.transducer = transducer;
            return this;
        }

        public Builder setParaformer(OfflineParaformerModelConfig paraformer) {
            this.paraformer = paraformer;
            return this;
        }

        public Builder setNemo(OfflineNemoEncDecCtcModelConfig nemo) {
            this.nemo = nemo;
            return this;
        }


        public Builder setTeleSpeech(String teleSpeech) {
            this.teleSpeech = teleSpeech;
            return this;
        }

        public Builder setWhisper(OfflineWhisperModelConfig whisper) {
            this.whisper = whisper;
            return this;
        }

        public Builder setTokens(String tokens) {
            this.tokens = tokens;
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

        public Builder setModelType(String modelType) {
            this.modelType = modelType;
            return this;
        }

        public void setModelingUnit(String modelingUnit) {
            this.modelingUnit = modelingUnit;
        }

        public void setBpeVocab(String bpeVocab) {
            this.bpeVocab = bpeVocab;
        }
    }
}