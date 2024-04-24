// Copyright 2024 Xiaomi Corporation
package com.k2fsa.sherpa.onnx;

public class OnlineRecognizerResult {
    private final String text;
    private final String[] tokens;
    private final float[] timestamps;

    public OnlineRecognizerResult(String text, String[] tokens, float[] timestamps) {
        this.text = text;
        this.tokens = tokens;
        this.timestamps = timestamps;
    }

    public String getText() {
        return text;
    }

    public String[] getTokens() {
        return tokens;
    }

    public float[] getTimestamps() {
        return timestamps;
    }
}
