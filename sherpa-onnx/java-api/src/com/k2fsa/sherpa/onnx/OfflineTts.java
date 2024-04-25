// Copyright 2024 Xiaomi Corporation

package com.k2fsa.sherpa.onnx;

public class OfflineTts {
    static {
        System.loadLibrary("sherpa-onnx-jni");
    }

    private long ptr = 0; // this is the asr engine ptrss

    public OfflineTts(OfflineTtsConfig config) {
        ptr = newFromFile(config);
    }

    public GeneratedAudio generate(String text) {
        return generate(text, 0, 1.0f);
    }

    public GeneratedAudio generate(String text, int sid) {
        return generate(text, sid, 1.0f);
    }

    public GeneratedAudio generate(String text, int sid, float speed) {
        Object[] arr = generateImpl(ptr, text, sid, speed);
        float[] samples = (float[]) arr[0];
        int sampleRate = (int) arr[1];
        return new GeneratedAudio(samples, sampleRate);
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

    private native void delete(long ptr);

    private native int getSampleRate(long ptr);

    private native int getNumSpeakers(long ptr);

    private native Object[] generateImpl(long ptr, String text, int sid, float speed);

    private native long newFromFile(OfflineTtsConfig config);
}
