// Copyright 2024 Xiaomi Corporation

package com.k2fsa.sherpa.onnx;

public class OfflinePunctuation {
    private long ptr = 0;

    public OfflinePunctuation(OfflinePunctuationConfig config) {
        LibraryLoader.maybeLoad();
        ptr = newFromFile(config);
        if (ptr == 0) {
            throw new IllegalArgumentException("Invalid OfflinePunctuationConfig: failed to create native OfflinePunctuation");
        }
    }

    public String addPunctuation(String text) {
        return addPunctuation(ptr, text);
    }

    @Override
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

    private native void delete(long ptr);

    private native long newFromFile(OfflinePunctuationConfig config);

    private native String addPunctuation(long ptr, String text);
}
