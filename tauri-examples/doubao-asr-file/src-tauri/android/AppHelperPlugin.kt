package com.plugin.apphelper

import android.app.Activity
import android.content.Intent
import android.net.Uri
import android.provider.OpenableColumns
import androidx.core.content.FileProvider
import app.tauri.annotation.Command
import app.tauri.annotation.InvokeArg
import app.tauri.annotation.TauriPlugin
import app.tauri.plugin.Invoke
import app.tauri.plugin.JSObject
import app.tauri.plugin.Plugin
import java.io.File

@InvokeArg
class DisplayNameArgs {
  lateinit var path: String
}

@InvokeArg
class ShareArgs {
  lateinit var path: String
  var mime: String = "text/plain"
  var text: String = ""
}

/**
 * Small Android helpers that no Tauri plugin provides correctly:
 *
 *  - getDisplayName: resolve the human-readable name of a content:// URI
 *    ("msf:1495" is really "1_lei-jun.wav") via OpenableColumns.
 *  - share: hand a file (or text fallback) to the system share sheet, e.g.
 *    WeChat. The stock tauri-plugin-share calls the wrong command name on
 *    Android and never resolves, so we do it here.
 */
@TauriPlugin
class AppHelperPlugin(private val activity: Activity) : Plugin(activity) {
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

  @Command
  fun share(invoke: Invoke) {
    try {
      val args = invoke.parseArgs(ShareArgs::class.java)
      val context = activity.applicationContext
      val intent = Intent(Intent.ACTION_SEND)
      intent.type = args.mime
      var shared = false
      try {
        val uri = FileProvider.getUriForFile(
          context,
          "${context.packageName}.fileprovider",
          File(args.path)
        )
        intent.putExtra(Intent.EXTRA_STREAM, uri)
        intent.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
        shared = true
      } catch (ex: Exception) {
        // FileProvider has no rule for this path: share the text instead.
        intent.type = "text/plain"
        intent.putExtra(Intent.EXTRA_TEXT, args.text)
      }
      val chooser = Intent.createChooser(intent, null)
      chooser.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
      context.startActivity(chooser)
      val ret = JSObject()
      ret.put("sharedAsFile", shared)
      invoke.resolve(ret)
    } catch (ex: Exception) {
      invoke.reject(ex.message ?: "share failed")
    }
  }
}
