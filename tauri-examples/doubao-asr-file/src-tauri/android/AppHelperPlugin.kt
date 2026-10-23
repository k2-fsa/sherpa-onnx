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

@InvokeArg
class NameArgs {
  lateinit var name: String
}

@InvokeArg
class SaveArgs {
  lateinit var name: String
  lateinit var content: String
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

  // ---- public Downloads folder (MediaStore, Android 10+) ----------------
  // Results live where users can find them (Files app), not in the private
  // app sandbox. MediaStore.Downloads exists from API 29.

  private fun downloadsUri(): Uri? {
    if (android.os.Build.VERSION.SDK_INT < 29) return null
    return android.provider.MediaStore.Downloads.EXTERNAL_CONTENT_URI
  }

  private fun findInDownloads(name: String): Uri? {
    val collection = downloadsUri() ?: return null
    activity.contentResolver.query(
      collection,
      arrayOf(android.provider.MediaStore.MediaColumns._ID),
      "${android.provider.MediaStore.MediaColumns.DISPLAY_NAME}=? AND " +
        "${android.provider.MediaStore.MediaColumns.RELATIVE_PATH}=?",
      arrayOf(name, "Download/"),
      null
    )?.use { c ->
      if (c.moveToFirst()) {
        return android.content.ContentUris.withAppendedId(collection, c.getLong(0))
      }
    }
    return null
  }

  @Command
  fun saveToDownloads(invoke: Invoke) {
    try {
      val args = invoke.parseArgs(SaveArgs::class.java)
      val collection = downloadsUri()
        ?: return invoke.reject("Saving to the public Downloads folder needs Android 10 or newer")
      // Remember any earlier result: it is replaced only AFTER the new file is
      // fully written, so a failed save never loses the previous transcript.
      val oldUri = findInDownloads(args.name)
      val values = android.content.ContentValues().apply {
        put(android.provider.MediaStore.MediaColumns.DISPLAY_NAME, args.name)
        put(
          android.provider.MediaStore.MediaColumns.MIME_TYPE,
          if (args.name.endsWith(".srt")) "application/x-subrip" else "text/plain"
        )
        put(android.provider.MediaStore.MediaColumns.RELATIVE_PATH, "Download")
        if (android.os.Build.VERSION.SDK_INT >= 29) {
          put(android.provider.MediaStore.MediaColumns.IS_PENDING, 1)
        }
      }
      val uri = activity.contentResolver.insert(collection, values)
        ?: return invoke.reject("Cannot create ${args.name} in Downloads")

      // A null stream is a failed save — never report success for it.
      try {
        val stream = activity.contentResolver.openOutputStream(uri)
          ?: throw Exception("cannot open ${args.name} for writing")
        stream.use { out -> out.write(args.content.toByteArray(Charsets.UTF_8)) }
      } catch (ex: Exception) {
        activity.contentResolver.delete(uri, null, null) // drop the incomplete entry
        return invoke.reject("Cannot write ${args.name} in Downloads: ${ex.message}")
      }

      if (android.os.Build.VERSION.SDK_INT >= 29) {
        val done = android.content.ContentValues()
        done.put(android.provider.MediaStore.MediaColumns.IS_PENDING, 0)
        activity.contentResolver.update(uri, done, null, null)
      }
      // The replacement is complete — only now remove the previous copy.
      oldUri?.let { activity.contentResolver.delete(it, null, null) }
      val ret = JSObject()
      ret.put("name", args.name)
      invoke.resolve(ret)
    } catch (ex: Exception) {
      invoke.reject(ex.message ?: "save failed")
    }
  }

  @Command
  fun readFromDownloads(invoke: Invoke) {
    try {
      val args = invoke.parseArgs(NameArgs::class.java)
      val uri = findInDownloads(args.name)
        ?: return invoke.reject("${args.name} does not exist in Downloads")
      val text = activity.contentResolver.openInputStream(uri)?.use { input ->
        input.readBytes().toString(Charsets.UTF_8)
      } ?: return invoke.reject("Cannot read ${args.name}")
      val ret = JSObject()
      ret.put("content", text)
      invoke.resolve(ret)
    } catch (ex: Exception) {
      invoke.reject(ex.message ?: "read failed")
    }
  }

  @Command
  fun existsInDownloads(invoke: Invoke) {
    try {
      val args = invoke.parseArgs(NameArgs::class.java)
      val ret = JSObject()
      ret.put("exists", findInDownloads(args.name) != null)
      invoke.resolve(ret)
    } catch (ex: Exception) {
      invoke.reject(ex.message ?: "query failed")
    }
  }

  @Command
  fun deleteFromDownloads(invoke: Invoke) {
    try {
      val args = invoke.parseArgs(NameArgs::class.java)
      val uri = findInDownloads(args.name)
      val deleted = if (uri != null) {
        activity.contentResolver.delete(uri, null, null) > 0
      } else {
        false
      }
      val ret = JSObject()
      ret.put("deleted", deleted)
      invoke.resolve(ret)
    } catch (ex: Exception) {
      invoke.reject(ex.message ?: "delete failed")
    }
  }
}
