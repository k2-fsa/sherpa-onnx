// Copyright 2024 Xiaomi Corporation

package com.k2fsa.sherpa.onnx;

public class OfflineRecognizerResult {
    private final String text;
    private final String[] tokens;
    private final float[] timestamps;
    private final String lang;
    private final String emotion;
    private final String event;
    private final float[] durations;
    // ysProbs[i] is the log probability for tokens[i]
    private final float[] ysProbs;
    // The decoded word IDs. Empty for greedy search decoding, non-empty when
    // an HLG graph is used, i.e. when ctcFstDecoderConfig.graph is set.
    private final int[] words;

    public OfflineRecognizerResult(String text, String[] tokens, float[] timestamps, String lang, String emotion, String event, float[] durations, float[] ysProbs, int[] words) {
        this.text = text;
        this.tokens = tokens;
        this.timestamps = timestamps;
        this.lang = lang;
        this.emotion = emotion;
        this.event = event;
        this.durations = durations;
        this.ysProbs = ysProbs;
        this.words = words;
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

    public float[] getDurations() {
        return durations;
    }

    public float[] getYsProbs() {
        return ysProbs;
    }

    public int[] getWords() {
        return words;
    }
}
