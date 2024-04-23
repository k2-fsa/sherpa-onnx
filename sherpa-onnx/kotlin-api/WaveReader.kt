// Copyright (c)  2023  Xiaomi Corporation
package com.k2fsa.sherpa.onnx

import android.content.res.AssetManager

class WaveReader {
    companion object {
        // Read a mono wave file asset
        // The returned array has two entries:
        //  - the first entry contains an 1-D float array
        //  - the second entry is the sample rate
        external fun readWaveFromAsset(
            assetManager: AssetManager,
            filename: String,
        ): Array<Any>

        // Read a mono wave file from disk
        // The returned array has two entries:
        //  - the first entry contains an 1-D float array
        //  - the second entry is the sample rate
        external fun readWaveFromFile(
            filename: String,
        ): Array<Any>

        init {
            System.loadLibrary("sherpa-onnx-jni")
        }
    }
}
