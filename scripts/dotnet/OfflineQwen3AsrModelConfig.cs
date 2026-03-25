/// Copyright (c)  2026  Xiaomi Corporation

using System.Runtime.InteropServices;

namespace SherpaOnnx
{

    [StructLayout(LayoutKind.Sequential)]
    public struct OfflineQwen3AsrModelConfig
    {
        public OfflineQwen3AsrModelConfig()
        {
            ConvFrontend = "";
            Encoder = "";
            Decoder = "";
            Tokenizer = "";
            MaxTotalLen = 512;
            MaxNewTokens = 128;
            Temperature = 1e-6F;
            TopP = 0.8F;
            Seed = 42;
        }

        [MarshalAs(UnmanagedType.LPStr)]
        public string ConvFrontend;

        [MarshalAs(UnmanagedType.LPStr)]
        public string Encoder;

        [MarshalAs(UnmanagedType.LPStr)]
        public string Decoder;

        [MarshalAs(UnmanagedType.LPStr)]
        public string Tokenizer;

        public int MaxTotalLen;
        public int MaxNewTokens;
        public float Temperature;
        public float TopP;
        public int Seed;
    }
}
