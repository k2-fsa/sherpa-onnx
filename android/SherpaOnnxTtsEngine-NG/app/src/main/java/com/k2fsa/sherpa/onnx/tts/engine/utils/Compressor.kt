package com.k2fsa.sherpa.onnx.tts.engine.utils

import com.k2fsa.sherpa.onnx.tts.engine.utils.CompressUtils.uncompress
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.isActive
import org.apache.commons.compress.archivers.ArchiveEntry
import org.apache.commons.compress.archivers.ArchiveInputStream
import org.apache.commons.compress.archivers.ArchiveOutputStream
import org.apache.commons.compress.archivers.tar.TarArchiveEntry
import org.apache.commons.compress.archivers.tar.TarArchiveInputStream
import org.apache.commons.compress.archivers.tar.TarArchiveOutputStream
import org.apache.commons.compress.archivers.zip.ZipArchiveEntry
import org.apache.commons.compress.archivers.zip.ZipArchiveInputStream
import org.apache.commons.compress.archivers.zip.ZipArchiveOutputStream
import org.apache.commons.compress.compressors.bzip2.BZip2CompressorInputStream
import org.apache.commons.compress.compressors.gzip.GzipCompressorInputStream
import org.apache.commons.compress.compressors.gzip.GzipCompressorOutputStream
import org.apache.commons.compress.compressors.xz.XZCompressorInputStream
import org.apache.commons.compress.compressors.xz.XZCompressorOutputStream
import java.io.File
import java.io.InputStream
import java.io.OutputStream
import kotlin.coroutines.coroutineContext

object CompressorFactory {
    val compressors = listOf(
        TarBzip2Compressor(),
        ZipCompressor(),
        TarGzipCompressor(),
        TarXzCompressor()
    )

    fun createCompressor(type: String): CompressorInterface? {
        for (compressor in compressors) {
            if (compressor.verifyType(type)) {
                return compressor
            }
        }
        return null
    }
}

interface CompressorInterface {
    fun verifyType(type: String): Boolean

    suspend fun uncompress(
        ins: InputStream,
        outputDir: String,
        progressListener: CompressUtils.ProgressListener
    )

    suspend fun compress(
        dir: String,
        ous: OutputStream,
        progressListener: CompressUtils.ProgressListener
    )
}

abstract class ImplCompressor<E : ArchiveEntry?>(open val extName: List<String>) :
    CompressorInterface {
    override fun verifyType(type: String): Boolean {
        return extName.any { it.lowercase() == type.lowercase() }
    }

    abstract fun archiveInputStream(ins: InputStream): ArchiveInputStream<*>
    open fun archiveOutputStream(outs: OutputStream): ArchiveOutputStream<E> {
        TODO()
    }

    override suspend fun uncompress(
        ins: InputStream,
        outputDir: String,
        progressListener: CompressUtils.ProgressListener
    ) {
        archiveInputStream(ins).use { arIn ->
            arIn.uncompress(outputDir, progressListener)
        }
    }

    override suspend fun compress(
        dir: String, ous: OutputStream, progressListener: CompressUtils.ProgressListener
    ) {
        archiveOutputStream(ous).use { it ->
            CompressUtils.deepGetFiles(File(dir)) { file ->
                if (!coroutineContext.isActive) throw CancellationException()

                val entry = it.createArchiveEntry(file, file.relativeTo(File(dir)).path)!!
                it.putArchiveEntry(entry)
                file.inputStream().use { ins ->
                    val buffer = ByteArray(DEFAULT_BUFFER_SIZE)
                    var len = 0
                    var bytes = 0
                    while (ins.read(buffer).also { len = it } != -1) {
                        if (!coroutineContext.isActive) throw CancellationException()

                        it.write(buffer, 0, len)
                        bytes += len
                        progressListener.onEntryProgress(entry.name, entry.size, bytes.toLong())
                    }
                }
                it.closeArchiveEntry()
            }

        }
    }
}

class TarBzip2Compressor : ImplCompressor<ArchiveEntry>(listOf("tar.bz2", "tbz2")) {
    override fun archiveInputStream(ins: InputStream): ArchiveInputStream<*> =
        TarArchiveInputStream(BZip2CompressorInputStream(ins))
}

class ZipCompressor : ImplCompressor<ZipArchiveEntry>(listOf("zip")) {
    override fun archiveInputStream(ins: InputStream): ArchiveInputStream<*> =
        ZipArchiveInputStream(ins)

    override fun archiveOutputStream(outs: OutputStream): ArchiveOutputStream<ZipArchiveEntry> {
        return ZipArchiveOutputStream(outs)
    }

//    override suspend fun compress(dir: String, ous: OutputStream) {
//        val file = File(dir)
//
//        ZipArchiveOutputStream(ous).use { zipOus ->
//            CompressUtils.deepGetFiles(file) {
//                if (!coroutineContext.isActive) throw CancellationException()
//
//                val relPath = it.relativeTo(file).path
//                Log.e("TAG", it.absolutePath)
//                val entry = zipOus.createArchiveEntry(it, relPath)
//                entry.size = it.length()
//                zipOus.putArchiveEntry(entry)
//                it.inputStream().use { ins ->
//                    ins.copyTo(zipOus)
//                }
//                zipOus.closeArchiveEntry()
//            }
//            zipOus.finish()
//        }
//    }
}

class TarGzipCompressor : ImplCompressor<TarArchiveEntry>(listOf("tar.gz", "tgz")) {
    override fun archiveInputStream(ins: InputStream): ArchiveInputStream<*> =
        TarArchiveInputStream(GzipCompressorInputStream(ins))

    override fun archiveOutputStream(outs: OutputStream): ArchiveOutputStream<TarArchiveEntry> =
        TarArchiveOutputStream(GzipCompressorOutputStream(outs))
}

class TarXzCompressor : ImplCompressor<TarArchiveEntry>(listOf("tar.xz", "txz")) {
    override fun archiveInputStream(ins: InputStream): ArchiveInputStream<*> =
        TarArchiveInputStream(XZCompressorInputStream(ins))

    override fun archiveOutputStream(outs: OutputStream): ArchiveOutputStream<TarArchiveEntry> =
        TarArchiveOutputStream(XZCompressorOutputStream(outs))
}