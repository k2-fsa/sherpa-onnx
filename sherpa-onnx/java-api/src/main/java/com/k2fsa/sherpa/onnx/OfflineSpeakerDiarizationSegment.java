// Copyright 2024 Xiaomi Corporation

package com.k2fsa.sherpa.onnx;

public class OfflineSpeakerDiarizationSegment {
    private final float start;
    private final float end;
    private final int speaker;

    public OfflineSpeakerDiarizationSegment(float start, float end, int speaker) {
        this.start = start;
        this.end = end;
        this.speaker = speaker;
    }

    public float getStart() {
        return start;
    }

    public float getEnd() {
        return end;
    }

    public int getSpeaker() {
        return speaker;
    }
}
