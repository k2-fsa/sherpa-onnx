// Copyright 2024 Xiaomi Corporation

package com.k2fsa.sherpa.onnx;

public class OfflineSpeakerDiarization {
    static {
        System.loadLibrary("sherpa-onnx-jni");
    }

    private long ptr = 0;

    public OfflineSpeakerDiarization(OfflineSpeakerDiarizationConfig config) {
        ptr = newFromFile(config);
    }

    public int getSampleRate() {
        return getSampleRate(ptr);
    }

    // Only config.clustering is used. All other fields are ignored
    public void setConfig(OfflineSpeakerDiarizationConfig config) {
        setConfig(ptr, config);
    }

    public OfflineSpeakerDiarizationSegment[] process(float[] samples) {
        return process(ptr, samples);
    }

    public OfflineSpeakerDiarizationSegment[] processWithCallback(float[] samples, OfflineSpeakerDiarizationCallback callback) {
        return processWithCallback(ptr, samples, callback, 0);
    }

    public OfflineSpeakerDiarizationSegment[] processWithCallback(float[] samples, OfflineSpeakerDiarizationCallback callback, long arg) {
        return processWithCallback(ptr, samples, callback, arg);
    }

    protected void finalize() throws Throwable {
        release();
    }

    // You'd better call it manually if it is not used anymore
    public void release() {
        if (this.ptr == 0) {
            return;
        }
        delete(this.ptr);
        this.ptr = 0;
    }

    private native int getSampleRate(long ptr);

    private native void delete(long ptr);

    private native long newFromFile(OfflineSpeakerDiarizationConfig config);

    private native void setConfig(long ptr, OfflineSpeakerDiarizationConfig config);

    private native OfflineSpeakerDiarizationSegment[] process(long ptr, float[] samples);

    private native OfflineSpeakerDiarizationSegment[] processWithCallback(long ptr, float[] samples, OfflineSpeakerDiarizationCallback callback, long arg);
}