// Copyright 2024 Xiaomi Corporation

package com.k2fsa.sherpa.onnx;

public class OnlineRecognizerResult {
    private final String text;
    private final String[] tokens;
    private final float[] timestamps;
    private final float[] ysProbs;

    public OnlineRecognizerResult(String text, String[] tokens, float[] timestamps, float[] ysProbs) {
        this.text = text;
        this.tokens = tokens;
        this.timestamps = timestamps;
        this.ysProbs = ysProbs;
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

    public float[] getYsProbs() {
        return ysProbs;
    }

}
