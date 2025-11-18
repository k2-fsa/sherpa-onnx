// Copyright 2025 Xiaomi Corporation

package com.k2fsa.sherpa.onnx;

public class QnnConfig {
    private final String backendLib;  // unused
    private final String contextBinary;
    private final String systemLib;

    private QnnConfig(Builder builder) {
        this.backendLib = builder.backendLib;
        this.contextBinary = builder.contextBinary;
        this.systemLib = builder.systemLib;
    }

    public static Builder builder() {
        return new Builder();
    }

    public String getBackendLib() {
        return backendLib;
    }

    public String getContextBinary() {
        return contextBinary;
    }

    public String getSystemLib() {
        return systemLib;
    }

    public static class Builder {
        private String backendLib = "";
        private String contextBinary = "";
        private String systemLib = "";

        public QnnConfig build() {
            return new QnnConfig(this);
        }

        public Builder setBackendLib(String backendLib) {
            this.backendLib = backendLib;
            return this;
        }

        public Builder setContextBinary(String contextBinary) {
            this.contextBinary = contextBinary;
            return this;
        }

        public Builder setSystemLib(String systemLib) {
            this.systemLib = systemLib;
            return this;
        }
    }
}
