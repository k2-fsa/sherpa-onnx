/// Copyright (c)  2025  Xiaomi Corporation (authors: Fangjun Kuang)

using System.Runtime.InteropServices;

namespace SherpaOnnx
{
    [StructLayout(LayoutKind.Sequential)]
    public struct OfflineTtsZipVoiceModelConfig
    {
        public OfflineTtsZipVoiceModelConfig()
        {
            Tokens = "";
            Encoder = "";
            Decoder = "";
            Vocoder = "";
            DataDir = "";
            Lexicon = "";

            FeatScale = 0.1F;
            Tshift = 0.5F;
            TargetRms = 0.1F;
            GuidanceScale = 1.0F;
        }
        [MarshalAs(UnmanagedType.LPStr)]
        public string Tokens;

        [MarshalAs(UnmanagedType.LPStr)]
        public string Encoder;

        [MarshalAs(UnmanagedType.LPStr)]
        public string Decoder;

        [MarshalAs(UnmanagedType.LPStr)]
        public string Vocoder;

        [MarshalAs(UnmanagedType.LPStr)]
        public string DataDir;

        [MarshalAs(UnmanagedType.LPStr)]
        public string Lexicon;

        public float FeatScale;
        public float Tshift;
        public float TargetRms;
        public float GuidanceScale;
    }
}
