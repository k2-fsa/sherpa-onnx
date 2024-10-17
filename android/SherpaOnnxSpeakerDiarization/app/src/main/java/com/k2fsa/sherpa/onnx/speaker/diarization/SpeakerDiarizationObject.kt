package com.k2fsa.sherpa.onnx.speaker.diarization

import android.content.res.AssetManager
import android.util.Log
import com.k2fsa.sherpa.onnx.FastClusteringConfig
import com.k2fsa.sherpa.onnx.OfflineSpeakerDiarization
import com.k2fsa.sherpa.onnx.OfflineSpeakerDiarizationConfig
import com.k2fsa.sherpa.onnx.OfflineSpeakerSegmentationModelConfig
import com.k2fsa.sherpa.onnx.OfflineSpeakerSegmentationPyannoteModelConfig
import com.k2fsa.sherpa.onnx.SpeakerEmbeddingExtractorConfig

// Please download
// https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2
// then unzip it, rename model.onnx to segmentation.onnx, and mv
// segmentation.onnx to the assets folder
val segmentationModel = "segmentation.onnx"

// please download it from
// https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx
// and rename it to embedding.onnx
// and move it to the assets folder
val embeddingModel = "embedding.onnx"

// in the end, your assets folder should look like below
/*
(py38) fangjuns-MacBook-Pro:assets fangjun$ pwd
/Users/fangjun/open-source/sherpa-onnx/android/SherpaOnnxSpeakerDiarization/app/src/main/assets
(py38) fangjuns-MacBook-Pro:assets fangjun$ ls -lh
total 89048
-rw-r--r--  1 fangjun  staff    38M Oct 12 20:28 embedding.onnx
-rw-r--r--  1 fangjun  staff   5.7M Oct 12 20:28 segmentation.onnx
 */

object SpeakerDiarizationObject {
    var _sd: OfflineSpeakerDiarization? = null
    val sd: OfflineSpeakerDiarization
        get() {
            return _sd!!
        }

    fun initSpeakerDiarization(assetManager: AssetManager? = null) {
        synchronized(this) {
            if (_sd != null) {
                return
            }
            Log.i(TAG, "Initializing sherpa-onnx speaker diarization")

            val config = OfflineSpeakerDiarizationConfig(
                segmentation = OfflineSpeakerSegmentationModelConfig(
                    pyannote = OfflineSpeakerSegmentationPyannoteModelConfig(
                        segmentationModel
                    ),
                    debug = true,
                ),
                embedding = SpeakerEmbeddingExtractorConfig(
                    model = embeddingModel,
                    debug = true,
                    numThreads = 2,
                ),
                clustering = FastClusteringConfig(numClusters = -1, threshold = 0.5f),
                minDurationOn = 0.2f,
                minDurationOff = 0.5f,
            )
            _sd = OfflineSpeakerDiarization(assetManager = assetManager, config = config)
        }
    }
}
