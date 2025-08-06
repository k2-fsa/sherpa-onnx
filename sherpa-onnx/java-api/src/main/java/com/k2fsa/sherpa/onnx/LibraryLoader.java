package com.k2fsa.sherpa.onnx;

import com.k2fsa.sherpa.onnx.utils.LibraryUtils;

public class LibraryLoader {
    private static volatile boolean autoLoadEnabled = true;
    private static volatile boolean isLoaded = false;

    static synchronized void loadLibrary() {
        if (!isLoaded) {
            //System.loadLibrary("sherpa-onnx-jni");
            LibraryUtils.load();
            isLoaded = true;
        }
    }

    public static void setAutoLoadEnabled(boolean enabled) {
        autoLoadEnabled = enabled;
    }

    static void maybeLoad() {
        if (autoLoadEnabled) {
            loadLibrary();
        }
    }
}