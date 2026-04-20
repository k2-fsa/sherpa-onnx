// Copyright 2024 Xiaomi Corporation

package com.k2fsa.sherpa.onnx;

import java.util.Arrays;

public class KeywordSpotterResult {
    private final String keyword;
    private final String[] tokens;
    private final float[] timestamps;

    public KeywordSpotterResult(String keyword, String[] tokens, float[] timestamps) {
        this.keyword = keyword;
        this.tokens = tokens;
        this.timestamps = timestamps;
    }

    public String getKeyword() {
        return keyword;
    }

    public String[] getTokens() {
        return tokens;
    }

    public float[] getTimestamps() {
        return timestamps;
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("Keyword: ").append(keyword).append("\n");
        sb.append("Tokens: ").append(Arrays.toString(tokens)).append("\n");
        sb.append("Timestamps: ").append(Arrays.toString(timestamps));
        return sb.toString();
    }
}
