/// Copyright (c)  2025  Xiaomi Corporation (authors: Fangjun Kuang)

using System.Runtime.InteropServices;

namespace SherpaOnnx
{
    [StructLayout(LayoutKind.Sequential)]
    public struct OfflineTtsKittenModelConfig
    {
        public OfflineTtsKittenModelConfig()
        {
            Model = "";
            Voices = "";
            Tokens = "";
            DataDir = "";
            Lexicon = "";

            LengthScale = 1.0F;
        }
        [MarshalAs(UnmanagedType.LPStr)]
        public string Model;

        [MarshalAs(UnmanagedType.LPStr)]
        public string Voices;

        [MarshalAs(UnmanagedType.LPStr)]
        public string Tokens;

        /// <summary>
        /// Deprecated. Ignored since v2.0.0. Use Lexicon or
        /// OfflineTtsGenerationConfig.PhonemeCodepoints instead.
        /// </summary>
        [MarshalAs(UnmanagedType.LPStr)]
        public string DataDir;

        public float LengthScale;

        [MarshalAs(UnmanagedType.LPStr)]
        public string Lexicon;
    }
}
