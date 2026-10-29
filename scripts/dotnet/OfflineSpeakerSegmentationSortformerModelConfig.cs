/// Copyright (c)  2026  Silvio Tomatis

using System.Runtime.InteropServices;

namespace SherpaOnnx
{

    /// Sortformer end-to-end diarization model, e.g., Nemotron-3-Diarization.
    /// If Model is set, pyannote, embedding and clustering are ignored.
    [StructLayout(LayoutKind.Sequential)]
    public struct OfflineSpeakerSegmentationSortformerModelConfig
    {
        public OfflineSpeakerSegmentationSortformerModelConfig()
        {
            Model = "";
            Threshold = 0.5f;
        }

        [MarshalAs(UnmanagedType.LPStr)]
        public string Model;
        public float Threshold;
    }
}
