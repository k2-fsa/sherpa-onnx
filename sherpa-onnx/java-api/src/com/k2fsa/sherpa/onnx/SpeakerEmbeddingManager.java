// Copyright 2024 Xiaomi Corporation

package com.k2fsa.sherpa.onnx;

public class SpeakerEmbeddingManager {
    static {
        System.loadLibrary("sherpa-onnx-jni");
    }

    private long ptr = 0;

    public SpeakerEmbeddingManager(int dim) {
        ptr = create(dim);
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

    public boolean add(String name, float[] embedding) {
        return add(ptr, name, embedding);
    }

    public boolean add(String name, float[][] embedding) {
        return addList(ptr, name, embedding);
    }

    public boolean remove(String name) {
        return remove(ptr, name);
    }

    public String search(float[] embedding, float threshold) {
        return search(ptr, embedding, threshold);
    }

    public boolean verify(String name, float[] embedding, float threshold) {
        return verify(ptr, name, embedding, threshold);
    }

    public boolean contains(String name) {
        return contains(ptr, name);
    }

    public int getNumSpeakers() {
        return numSpeakers(ptr);
    }

    public String[] getAllSpeakerNames() {
        return allSpeakerNames(ptr);
    }

    private native long create(int dim);

    private native void delete(long ptr);

    private native boolean add(long ptr, String name, float[] embedding);

    private native boolean addList(long ptr, String name, float[][] embedding);

    private native boolean remove(long ptr, String name);

    private native String search(long ptr, float[] embedding, float threshold);

    private native boolean verify(long ptr, String name, float[] embedding, float threshold);

    private native boolean contains(long ptr, String name);

    private native int numSpeakers(long ptr);

    private native String[] allSpeakerNames(long ptr);
}
