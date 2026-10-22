package com.plugin.displayname

import android.app.Activity
import android.net.Uri
import android.provider.OpenableColumns
import app.tauri.annotation.Command
import app.tauri.annotation.InvokeArg
import app.tauri.annotation.TauriPlugin
import app.tauri.plugin.Invoke
import app.tauri.plugin.JSObject
import app.tauri.plugin.Plugin

@InvokeArg
class DisplayNameArgs {
  lateinit var path: String
}

/**
 * Resolves the human-readable file name of a content:// URI via
 * OpenableColumns.DISPLAY_NAME. The system document picker hands back URIs
 * like "msf:1495" that carry no name at all — this is the only way to learn
 * that the file is really "1_lei-jun.wav".
 */
@TauriPlugin
class DisplayNamePlugin(private val activity: Activity) : Plugin(activity) {
  @Command
  fun getDisplayName(invoke: Invoke) {
    try {
      val args = invoke.parseArgs(DisplayNameArgs::class.java)
      val uri = Uri.parse(args.path)
      var name: String? = null
      activity.contentResolver
        .query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)
        ?.use { c ->
          if (c.moveToFirst()) {
            val idx = c.getColumnIndex(OpenableColumns.DISPLAY_NAME)
            if (idx >= 0) name = c.getString(idx)
          }
        }
      if (name.isNullOrEmpty()) name = uri.lastPathSegment
      val ret = JSObject()
      ret.put("name", name ?: "")
      invoke.resolve(ret)
    } catch (ex: Exception) {
      invoke.reject(ex.message ?: "failed to query display name")
    }
  }
}
