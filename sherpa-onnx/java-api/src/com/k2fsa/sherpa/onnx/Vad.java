// Copyright 2024 Xiaomi Corporation

package com.k2fsa.sherpa.onnx;

public class Vad {
    static {
        System.loadLibrary("sherpa-onnx-jni");
    }

    private long ptr = 0;

    public Vad(VadModelConfig config) {
        ptr = newFromFile(config);
    }

    @Override
    protected void finalize() throws Throwable {
        release();
    }

    public void release() {
        if (this.ptr == 0) {
            return;
        }
        delete(this.ptr);
        this.ptr = 0;
    }

    public void acceptWaveform(float[] samples) {
        acceptWaveform(this.ptr, samples);
    }

    public boolean empty() {
        return empty(this.ptr);
    }

    public void pop() {
        pop(this.ptr);
    }

    public void clear() {
        clear(this.ptr);
    }

    public void reset() {
        reset(this.ptr);
    }

    public SpeechSegment front() {
        Object[] arr = front(this.ptr);
        int start = (int) arr[0];
        float[] samples = (float[]) arr[1];

        return new SpeechSegment(start, samples);
    }

    public boolean isSpeechDetected() {
        return isSpeechDetected(this.ptr);
    }

    private native void delete(long ptr);

    private native long newFromFile(VadModelConfig config);

    private native void acceptWaveform(long ptr, float[] samples);

    private native boolean empty(long ptr);

    private native void pop(long ptr);

    private native void clear(long ptr);

    private native Object[] front(long ptr);

    private native boolean isSpeechDetected(long ptr);

    private native void reset(long ptr);
}
