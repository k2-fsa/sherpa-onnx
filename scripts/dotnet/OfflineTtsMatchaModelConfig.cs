/// Copyright (c)  2025  Xiaomi Corporation (authors: Fangjun Kuang)

using System.Runtime.InteropServices;

namespace SherpaOnnx
{
    [StructLayout(LayoutKind.Sequential)]
    public struct OfflineTtsMatchaModelConfig
    {
        public OfflineTtsMatchaModelConfig()
        {
            AcousticModel = "";
            Vocoder = "";
            Lexicon = "";
            Tokens = "";
            DataDir = "";

            NoiseScale = 0.667F;
            LengthScale = 1.0F;

            DictDir = "";
        }
        [MarshalAs(UnmanagedType.LPStr)]
        public string AcousticModel;

        [MarshalAs(UnmanagedType.LPStr)]
        public string Vocoder;

        [MarshalAs(UnmanagedType.LPStr)]
        public string Lexicon;

        [MarshalAs(UnmanagedType.LPStr)]
        public string Tokens;

        [MarshalAs(UnmanagedType.LPStr)]
        public string DataDir;

        public float NoiseScale;
        public float LengthScale;

        [MarshalAs(UnmanagedType.LPStr)]
        public string DictDir;
    }
}
