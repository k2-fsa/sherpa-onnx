/*
 * // Copyright 2022-2023 by zhaoming
 */

package com.k2fsa.sherpa.onnx;

public class OnlineTransducerModelConfig {
    private final String encoder;
    private final String decoder;
    private final String joiner;

    public OnlineTransducerModelConfig(String encoder, String decoder, String joiner) {
        this.encoder = encoder;
        this.decoder = decoder;
        this.joiner = joiner;
    }

    public String getEncoder() {
        return encoder;
    }

    public String getDecoder() {
        return decoder;
    }

    public String getJoiner() {
        return joiner;
    }
}
