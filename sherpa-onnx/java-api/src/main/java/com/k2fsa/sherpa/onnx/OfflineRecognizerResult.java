// Copyright 2024 Xiaomi Corporation

package com.k2fsa.sherpa.onnx;

public class OfflineRecognizerResult {
    private final String text;
    private final String[] tokens;
    private final float[] timestamps;
    private final String lang;
    private final String emotion;
    private final String event;

    public OfflineRecognizerResult(String text, String[] tokens, float[] timestamps, String lang, String emotion, String event) {
        this.text = text;
        this.tokens = tokens;
        this.timestamps = timestamps;
        this.lang = lang;
        this.emotion = emotion;
        this.event = event;
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

    public String getLang() {
        return lang;
    }

    public String getEmotion() {
        return emotion;
    }

    public String getEvent() {
        return event;
    }
}
