// Copyright 2026 Matias Lin

package com.k2fsa.sherpa.onnx;

public class OfflineDiacritization {
    private long ptr = 0;

    public OfflineDiacritization(OfflineDiacritizationConfig config) {
        LibraryLoader.maybeLoad();
        ptr = newFromFile(config);
        if (ptr == 0) {
            throw new IllegalArgumentException("Invalid OfflineDiacritizationConfig: failed to create native OfflineDiacritization");
        }
    }

    public String addDiacritics(String text) {
        return addDiacritics(ptr, text);
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

    private native long newFromFile(OfflineDiacritizationConfig config);

    private native String addDiacritics(long ptr, String text);
}
