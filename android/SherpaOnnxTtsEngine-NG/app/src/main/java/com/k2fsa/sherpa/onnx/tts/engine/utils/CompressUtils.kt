package com.k2fsa.sherpa.onnx.tts.engine.utils

import android.util.Log
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.isActive
import kotlinx.coroutines.withContext
import org.apache.commons.compress.archivers.ArchiveEntry
import org.apache.commons.compress.archivers.ArchiveInputStream
import org.apache.commons.io.FileUtils
import java.io.File
import kotlin.coroutines.coroutineContext


object CompressUtils {
    const val TAG = "CompressUtils"

    fun interface ProgressListener {
        fun onEntryProgress(name: String, entrySize: Long, bytes: Long)
    }

    suspend fun ArchiveInputStream<*>.uncompress(
        outputDir: String,
        progressListener: ProgressListener
    ) {
        createFile(outputDir, "").mkdirs()
        var totalBytes = 0L

        var entry: ArchiveEntry
        try {
            while (nextEntry.also { entry = it } != null) {
                totalBytes += entry.size
                val file = createFile(outputDir, entry.name)

                if (entry.isDirectory) {
                    file.mkdirs()
                } else {
                    withContext(Dispatchers.IO) {
                        if (file.exists()) {
                            file.delete()
                        } else {
                            FileUtils.createParentDirectories(file)
                            file.createNewFile()
                        }
                    }

                    file.outputStream().use { out ->
                        val buffer = ByteArray(DEFAULT_BUFFER_SIZE)
                        var bytes = 0L
                        var len = 0
                        while (read(buffer).also { len = it } != -1) {
                            if (!coroutineContext.isActive) {
                                throw CancellationException()
                            }

                            out.write(buffer, 0, len)
                            bytes += len
                            progressListener.onEntryProgress(entry.name, entry.size, bytes)
                        }
                    }
                }
            }
        } catch (_: NullPointerException) {
        }
    }


    private fun createFile(outputDir: String, name: String): File {
        return File(outputDir + File.separator + name)
    }

    suspend fun deepGetFiles(file: File, onFile: suspend (File) -> Unit) {
        if (file.isDirectory) {
            file.listFiles()?.forEach {
                deepGetFiles(it, onFile)
            }
        } else {
            onFile(file)
        }
    }
}