package com.k2fsa.sherpa.onnx;

public class SpeechSegment {

    private final int start;
    private final float[] samples;

    public SpeechSegment(int start, float[] samples) {
        this.start = start;
        this.samples = samples;
    }

    public int getStart() {
        return start;
    }

    public float[] getSamples() {
        return samples;
    }
}
