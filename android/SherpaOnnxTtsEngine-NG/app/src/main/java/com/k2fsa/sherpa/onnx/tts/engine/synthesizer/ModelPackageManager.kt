package com.k2fsa.sherpa.onnx.tts.engine.synthesizer

import com.drake.net.Get
import com.drake.net.component.Progress
import com.drake.net.exception.ResponseException
import com.drake.net.interfaces.ProgressListener
import com.k2fsa.sherpa.onnx.tts.engine.AppConst
import com.k2fsa.sherpa.onnx.tts.engine.FileConst
import com.k2fsa.sherpa.onnx.tts.engine.GithubRelease
import com.k2fsa.sherpa.onnx.tts.engine.synthesizer.config.Model
import com.k2fsa.sherpa.onnx.tts.engine.utils.CompressUtils
import com.k2fsa.sherpa.onnx.tts.engine.utils.CompressorFactory
import kotlinx.coroutines.coroutineScope
import kotlinx.serialization.ExperimentalSerializationApi
import kotlinx.serialization.json.decodeFromStream
import okhttp3.Response
import org.apache.commons.io.FileUtils
import java.io.File
import java.io.InputStream
import java.io.OutputStream
import java.util.UUID

object ModelPackageManager {
    val supportedTypes = listOf(
        "tar.bz2",
        "tgz",
        "tar.gz",
        "txz",
        "tar.xz",
        "zip"
    )

    init {
        runCatching {
            clearCache()
        }
    }

    private fun clearCache() {
        FileUtils.deleteDirectory(File(FileConst.cacheModelDir))
        FileUtils.deleteDirectory(File(FileConst.cacheDownloadDir))
    }

    @OptIn(ExperimentalSerializationApi::class)
    suspend fun getTtsModels(): List<GithubRelease.Asset> = coroutineScope {
        val resp = Get<Response>("").await()

        val body = resp.body
        return@coroutineScope if (resp.isSuccessful && body != null)
            AppConst.jsonBuilder.decodeFromStream(body.byteStream())
        else {
            throw ResponseException(resp, "Failed to get tts models, ${resp.code} ${resp.message}")
        }
    }

    /**
     * Delete directory: [FileConst.modelDir]/[modelId]
     */
    fun deleteModel(modelId: String): Boolean {
        val dir = File(FileConst.modelDir + File.separator + modelId)
        try {
            FileUtils.deleteDirectory(dir)
        } catch (_: Exception) {
            return false
        }

        return true
    }

    /**
     * Unzip package to [FileConst.cacheModelDir]/[subDir]`
     *
     * [subDir] default to UUID
     */
    suspend fun extractModelPackage(
        type: String,
        ins: InputStream,
        progressListener: CompressUtils.ProgressListener,
        subDir: String = UUID.randomUUID().toString(),
    ): String {
        val target = FileConst.cacheModelDir + File.separator + subDir

        val compressor = CompressorFactory.createCompressor(type) ?: throw IllegalArgumentException(
            "Unsupported type: $type"
        )
        compressor.uncompress(ins, target, progressListener)

        return target
    }

    /**
     * Install model package from local directory
     *
     * [source] example: [FileConst.cacheModelDir]/$uuid`
     */
    fun installModelPackageFromDir(source: File): Boolean {
        val dirs = source.listFiles { file, _ ->
            file.isDirectory
        } ?: return false

        val models = dirs.mapNotNull {
            ConfigModelManager.analyzeToModel(it)
        }

        val ok = models.isNotEmpty()
        if (ok) {
            FileUtils.copyDirectory(source, File(FileConst.modelDir))
            ConfigModelManager.addModel(*models.toTypedArray())
        }

        return ok
    }

    /**
     * Auto extract and install model package from input stream
     *
     * [ins] *.tar.bz2 input stream
     */
    suspend fun installPackage(
        type: String,
        ins: InputStream,
        onUnzipProgress: (file: String, total: Long, current: Long) -> Unit,
        onStartMoveFiles: () -> Unit
    ): Boolean {
        val target = extractModelPackage(type, ins, progressListener = { name, entrySize, bytes ->
            onUnzipProgress(name, entrySize, bytes)
        })

        onStartMoveFiles()
        return installModelPackageFromDir(File(target))
    }

    /**
     * Download model package from url and install
     */
    suspend fun installPackageFromUrl(
        url: String,
        fileName: String,
        onDownloadProgress: (Progress) -> Unit,
        onUnzipProgress: (file: String, total: Long, current: Long) -> Unit,
        onStartMoveFiles: () -> Unit
    ): Boolean {
        val file = downloadModelPackage(url, fileName) {
            onDownloadProgress(it)
        }

        val type = fileName.substringAfter(".")

        return file.inputStream().use {
            installPackage(type, it, onUnzipProgress, onStartMoveFiles)
        }
    }

    private suspend fun downloadModelPackage(
        url: String,
        fileName: String,
        onProgress: (Progress) -> Unit
    ): File = coroutineScope {
        val downloadDir = File(FileConst.cacheDownloadDir)
        downloadDir.mkdirs()
        val file = Get<File>(url) {
            if (fileName.isNotBlank())
                setDownloadFileName(fileName)
            setDownloadDir(downloadDir)
            addDownloadListener(object : ProgressListener() {
                override fun onProgress(p: Progress) {
                    onProgress(p)
                }
            })
        }.await()

        return@coroutineScope file
    }


    suspend fun exportModelsToZip(
        models: List<String>,
        type: String,
        ous: OutputStream,
        onZipProgress: CompressUtils.ProgressListener
    ): File {
        val cacheDir = File(FileConst.cacheModelDir + File.separator + UUID.randomUUID().toString())
        cacheDir.mkdirs()

        models.forEach {
            val modelDir = File(FileConst.modelDir + File.separator + it)
            val target = File(cacheDir, it)
            FileUtils.copyDirectory(modelDir, target)
        }

        val compressor = CompressorFactory.createCompressor(type) ?: throw IllegalArgumentException(
            "Unsupported type: $type"
        )

        compressor.compress(cacheDir.absolutePath, ous, onZipProgress)

        return File("")
    }
}