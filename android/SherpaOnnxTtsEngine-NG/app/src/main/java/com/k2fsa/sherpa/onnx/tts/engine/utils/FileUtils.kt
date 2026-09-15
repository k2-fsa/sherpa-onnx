package com.k2fsa.sherpa.onnx.tts.engine.utils

import android.content.Context
import android.net.Uri
import androidx.documentfile.provider.DocumentFile
import java.io.File


fun Uri.fileName(context: Context): String {
    var name: String? = null
    scheme?.let {
        when (it) {
            "content" -> {
                name = DocumentFile.fromSingleUri(context, this)?.name
            }

            "file" -> {
                path?.let {
                    name = File(it).name
                }
            }

            else -> {}
        }
    }

    return name ?: ""
}