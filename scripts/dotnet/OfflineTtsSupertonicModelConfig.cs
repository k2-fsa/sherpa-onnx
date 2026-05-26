/// Copyright (c)  2026  Xiaomi Corporation (authors: Fangjun Kuang)

using System.Runtime.InteropServices;

namespace SherpaOnnx
{
    [StructLayout(LayoutKind.Sequential)]
    public struct OfflineTtsSupertonicModelConfig
    {
        public OfflineTtsSupertonicModelConfig()
        {
            DurationPredictor = "";
            TextEncoder = "";
            VectorEstimator = "";
            Vocoder = "";
            TtsJson = "";
            UnicodeIndexer = "";
            VoiceStyle = "";
        }

        [MarshalAs(UnmanagedType.LPStr)]
        public string DurationPredictor;

        [MarshalAs(UnmanagedType.LPStr)]
        public string TextEncoder;

        [MarshalAs(UnmanagedType.LPStr)]
        public string VectorEstimator;

        [MarshalAs(UnmanagedType.LPStr)]
        public string Vocoder;

        [MarshalAs(UnmanagedType.LPStr)]
        public string TtsJson;

        [MarshalAs(UnmanagedType.LPStr)]
        public string UnicodeIndexer;

        [MarshalAs(UnmanagedType.LPStr)]
        public string VoiceStyle;
    }
}
