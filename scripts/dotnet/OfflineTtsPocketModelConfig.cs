/// Copyright (c)  2026  Xiaomi Corporation (authors: Fangjun Kuang)

using System.Runtime.InteropServices;

namespace SherpaOnnx
{
    [StructLayout(LayoutKind.Sequential)]
    public struct OfflineTtsPocketModelConfig
    {
        // Default constructor for convenience
        public OfflineTtsPocketModelConfig()
        {
            LmFlow = "";
            LmMain = "";
            Encoder = "";
            Decoder = "";
            TextConditioner = "";
            VocabJson = "";
            TokenScoresJson = "";
            VoiceEmbeddingCacheCapacity = 50;
        }

        [MarshalAs(UnmanagedType.LPStr)]
        public string LmFlow;

        [MarshalAs(UnmanagedType.LPStr)]
        public string LmMain;

        [MarshalAs(UnmanagedType.LPStr)]
        public string Encoder;

        [MarshalAs(UnmanagedType.LPStr)]
        public string Decoder;

        [MarshalAs(UnmanagedType.LPStr)]
        public string TextConditioner;

        [MarshalAs(UnmanagedType.LPStr)]
        public string VocabJson;

        [MarshalAs(UnmanagedType.LPStr)]
        public string TokenScoresJson;

        public int VoiceEmbeddingCacheCapacity;
    }
}

