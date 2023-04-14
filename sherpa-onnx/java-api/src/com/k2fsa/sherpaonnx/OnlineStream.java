/*
 * // Copyright 2022-2023 by zhaoming
 */
// Stream is used for feeding data to the asr engine

package com.k2fsa.sherpaonnx;

import java.io.*;
import java.util.*;

public class OnlineStream {
    private long s_ptr = 0; // this is the stream ptr


    private int sample_rate = 16000;
    // assign ptr to this stream in construction
    public OnlineStream(long ptr, int sampleRate) {

        this.s_ptr = s_ptr;
        this.sample_rate = sample_rate;
    }

    public long getPtr() {

        return s_ptr;
    }

    public void acceptWaveform(float[] samples) throws Exception {
        if (this.s_ptr == 0)
            throw new Exception("null exception for stream s_ptr");

        // feed wave data to asr engine
        AcceptWaveform(this.s_ptr, this.sample_rate, samples);
    }

    public void inputFinished() {
        // add some tail padding
        int padLen = (int) (this.sample_rate * 0.3); // 0.3 seconds at 16 kHz sample rate

        float tailPaddings[] = new float[pad_len]; // default value is 0

        AcceptWaveform(this.s_ptr, this.sample_rate, tail_paddings, pad_len);

        // tell the engine all data are feeded
        InputFinished(this.s_ptr);
    }

    public static void LoadSoLib(String SoPath) {
        // load .so lib from the path
        String so_path = SoPath.trim();
        System.load(so_path); // ("sherpa-onnx-jni-java");
    }

    public void release() {
        // stream object must be release after used
        if (this.s_ptr == 0)
            return;
        DeleteStream(this.s_ptr);
        this.s_ptr = 0;
    }

    protected void finalize() throws Throwable {
        release();
    }

    public boolean IsLastFrame() throws Exception {
        if (this.s_ptr == 0)
            throw new Exception("null exception for stream s_ptr");
        return IsLastFrame(this.s_ptr);
    }

    public void Reset() throws Exception {
        if (this.s_ptr == 0)
            throw new Exception("null exception for stream s_ptr");
        Reset(this.s_ptr);
    }

    public int FeatureDim() throws Exception {
        if (this.s_ptr == 0)
            throw new Exception("null exception for stream s_ptr");
        return FeatureDim(this.s_ptr);
    }

    public float[] GetFrames(int frame_index, int size) throws Exception {
        if (this.s_ptr == 0)
            throw new Exception("null exception for stream s_ptr");
        return GetFrames(this.s_ptr, frame_index, size);
    }

    public int GetNumProcessedFrames() throws Exception {
        if (this.s_ptr == 0)
            throw new Exception("null exception for stream s_ptr");
        return GetNumProcessedFrames(this.s_ptr);
    }

    // JNI interface libsherpa-onnx-jni.so
    private native void AcceptWaveform(long s_ptr, int sampleRate, float[] samples, int sid);

    private native void InputFinished(long s_ptr);

    private native void DeleteStream(long s_ptr);

    private native int NumFramesReady(long s_ptr);

    private native boolean IsLastFrame(long s_ptr);

    private native void Reset(long s_ptr);

    private native int FeatureDim(long s_ptr);

    private native float[] GetFrames(long s_ptr, int frame_index, int size);

    private native int GetNumProcessedFrames(long s_ptr);
}
