package com.k2fsa.sherpa.onnx.fa

import android.content.Context
import android.content.Intent
import android.net.Uri
import androidx.core.content.FileProvider
import java.io.File

/** Returns the display name of a content:// [uri], or a fallback. */
fun Context.queryDisplayName(uri: Uri): String {
    contentResolver.query(uri, null, null, null, null)?.use { cursor ->
        val index = cursor.getColumnIndex(android.provider.OpenableColumns.DISPLAY_NAME)
        if (index >= 0 && cursor.moveToFirst()) {
            cursor.getString(index)?.let { return it }
        }
    }
    return uri.lastPathSegment ?: "input"
}

/**
 * Writes [content] to a cache file named [fileName] and opens the system
 * share sheet (WeChat, email, ...). [fileName] should use a suffix the
 * receiving app accepts, e.g. ".TextGrid.txt" for a Praat TextGrid.
 */
fun Context.shareTextAsFile(content: String, fileName: String, mimeType: String = "text/plain") {
    val dir = File(cacheDir, "shared")
    dir.mkdirs()
    val file = File(dir, fileName)
    file.writeText(content)
    val uri = FileProvider.getUriForFile(this, "${packageName}.fileprovider", file)
    val intent = Intent(Intent.ACTION_SEND).apply {
        type = mimeType
        putExtra(Intent.EXTRA_STREAM, uri)
        addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
    }
    startActivity(Intent.createChooser(intent, "Share $fileName"))
}
