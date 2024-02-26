/*
 * // Copyright 2022-2023 by zhaoming
 */
// Stream is used for feeding data to the asr engine
package com.k2fsa.sherpa.onnx;

public class OnlineStream {
    private long ptr = 0; // this is the  stream ptr

    private int sampleRate = 16000;

    // assign ptr to this stream in construction
    public OnlineStream(long ptr, int sampleRate) {
        this.ptr = ptr;
        this.sampleRate = sampleRate;
    }

    public static void loadSoLib(String soPath) {
        // load .so lib from the path
        System.load(soPath.trim()); // ("sherpa-onnx-jni-java");
    }

    public long getPtr() {
        return ptr;
    }

    public void acceptWaveform(float[] samples) throws Exception {
        if (this.ptr == 0) throw new Exception("null exception for stream ptr");

        // feed wave data to asr engine
        acceptWaveform(this.ptr, this.sampleRate, samples);
    }

    public void inputFinished() {
        // add some tail padding
        int padLen = (int) (this.sampleRate * 0.3); // 0.3 seconds at 16 kHz sample rate
        float[] tailPaddings = new float[padLen]; // default value is 0
        acceptWaveform(this.ptr, this.sampleRate, tailPaddings);

        // tell the engine all data are feeded
        inputFinished(this.ptr);
    }

    public void release() {
        // stream object must be release after used
        if (this.ptr == 0) return;
        deleteStream(this.ptr);
        this.ptr = 0;
    }

    protected void finalize() throws Throwable {
        release();
    }

    public boolean isLastFrame() throws Exception {
        if (this.ptr == 0) throw new Exception("null exception for stream ptr");
        return isLastFrame(this.ptr);
    }

    public void reSet() throws Exception {
        if (this.ptr == 0) throw new Exception("null exception for stream ptr");
        reSet(this.ptr);
    }

    public int featureDim() throws Exception {
        if (this.ptr == 0) throw new Exception("null exception for stream ptr");
        return featureDim(this.ptr);
    }

    // JNI interface libsherpa-onnx-jni.so
    private native void acceptWaveform(long ptr, int sampleRate, float[] samples);

    private native void inputFinished(long ptr);

    private native void deleteStream(long ptr);

    private native int numFramesReady(long ptr);

    private native boolean isLastFrame(long ptr);

    private native void reSet(long ptr);

    private native int featureDim(long ptr);
}
