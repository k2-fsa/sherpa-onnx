// Copyright 2022-2023 by zhaoming
// Copyright 2024 Xiaomi Corporation
package com.k2fsa.sherpa.onnx;

public class OnlineStream {
    static {
        System.loadLibrary("sherpa-onnx-jni");
    }

    private long ptr = 0;

    public OnlineStream() {
        this.ptr = 0;
    }

    public OnlineStream(long ptr) {
        this.ptr = ptr;
    }

    public long getPtr() {
        return ptr;
    }

    public void setPtr(long ptr) {
        this.ptr = ptr;
    }

    public void acceptWaveform(float[] samples, int sampleRate) {
        acceptWaveform(this.ptr, samples, sampleRate);
    }

    public void inputFinished() {
        inputFinished(this.ptr);
    }

    public void release() {
        // stream object must be release after used
        if (this.ptr == 0) {
            return;
        }
        delete(this.ptr);
        this.ptr = 0;
    }

    @Override
    protected void finalize() throws Throwable {
        release();
        super.finalize();
    }

    private native void acceptWaveform(long ptr, float[] samples, int sampleRate);

    private native void inputFinished(long ptr);

    private native void delete(long ptr);
}