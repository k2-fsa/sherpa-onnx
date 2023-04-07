/*
 * // Copyright 2022-2023 by zhaoming
 */
// Stream is used for feed data to asr engine
package com.k2fsa.sherpaonnx;

import java.io.*;
import java.util.*;

public class OnlineStream {

    private long s_ptr; // this is the  stream ptr

    // assign ptr to this stream in construction
    public OnlineStream(long s_ptr) {
        this.s_ptr = s_ptr;

    }

    public long getS_ptr() {
        return s_ptr;
    }

    public void acceptWaveform(float[] samples, int sampleRate) {
        // feed wave data to asr engine
        AcceptWaveform(this.s_ptr, sampleRate, samples, samples.length);
    }

    public void inputFinished() {
        // tell the engine all data are feeded
        InputFinished(this.s_ptr);
    }

    public static void LoadSoLib(String SoPath) {

        // load .so lib from the path
        String so_path = SoPath.trim();
        System.load(so_path); // ("sherpa-onnx-jni-java");

    }
    public void release()
    {    
	    //stream object must be release after used
	    DeleteStream(this.s_ptr);
    }


    protected void finalize() throws Throwable {
        DeleteStream(this.s_ptr);
    }

    public boolean IsLastFrame()
    {
        return IsLastFrame(this.s_ptr);
    }

    public void Reset()
    {
        Reset(this.s_ptr);
    }

    public int FeatureDim()
    {
        return FeatureDim(this.s_ptr);
    }

    public float[] GetFrames(int frame_index,int size)
    {
        return GetFrames(this.s_ptr,frame_index,size);
    }

    public int GetNumProcessedFrames()
    {
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

    private native float[] GetFrames(long s_ptr,int frame_index,int size);

    private native int GetNumProcessedFrames(long s_ptr);
    
}
