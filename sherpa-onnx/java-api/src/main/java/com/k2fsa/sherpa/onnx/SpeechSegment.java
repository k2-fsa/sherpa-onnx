package com.k2fsa.sherpa.onnx;

public class SpeechSegment {

    private final long start;
    private final float[] samples;

    public SpeechSegment(long start, float[] samples) {
        this.start = start;
        this.samples = samples;
    }

    public long getStart() {
        return start;
    }

    public float[] getSamples() {
        return samples;
    }
}
