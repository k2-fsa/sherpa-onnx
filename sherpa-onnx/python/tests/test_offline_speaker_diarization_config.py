# Copyright (c)  2026  Silvio Tomatis

import tempfile
import unittest

import _sherpa_onnx as sherpa_onnx


@unittest.skipUnless(
    getattr(sherpa_onnx, "OfflineSpeakerSegmentationSortformerModelConfig", None)
    is not None,
    "Speaker diarization is disabled",
)
class TestOfflineSpeakerDiarizationConfig(unittest.TestCase):
    def test_pyannote_positional_constructor(self):
        pyannote = sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
            "pyannote.onnx", 0.2
        )
        config = sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote, 3, True, "cpu"
        )
        self.assertEqual(config.pyannote.model, "pyannote.onnx")
        self.assertEqual(config.num_threads, 3)
        self.assertTrue(config.debug)
        self.assertEqual(config.sortformer.model, "")

    def test_sortformer_needs_no_embedding_or_clustering(self):
        # Validation checks paths; no model is loaded here.
        with tempfile.NamedTemporaryFile() as model:
            segmentation = sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
                sortformer=sherpa_onnx.OfflineSpeakerSegmentationSortformerModelConfig(
                    model.name
                )
            )
            config = sherpa_onnx.OfflineSpeakerDiarizationConfig(segmentation)
            self.assertTrue(config.validate())
            # Sortformer takes precedence even when pyannote and embedding
            # paths or clustering parameters would be invalid.
            config.segmentation.pyannote.model = "missing-pyannote.onnx"
            config.embedding.model = "missing-embedding.onnx"
            config.clustering.num_clusters = -10
            self.assertTrue(config.validate())
            for threshold in (0, 1, -0.1, float("nan"), float("inf")):
                config.segmentation.sortformer.threshold = threshold
                self.assertFalse(config.validate())


if __name__ == "__main__":
    unittest.main()
