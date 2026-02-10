/// Copyright (c)  2025  Xiaomi Corporation (authors: Fangjun Kuang)

using System.Runtime.InteropServices;

namespace SherpaOnnx
{

    [StructLayout(LayoutKind.Sequential)]
    public struct OfflineFunAsrNanoModelConfig
    {
        public OfflineFunAsrNanoModelConfig()
        {
            EncoderAdaptor = "";
            LLM = "";
            Embedding = "";
            Tokenizer = "";
            SystemPrompt = "You are a helpful assistant.";
            UserPrompt = "语音转写：";
            MaxNewTokens = 512;
            Temperature = 1e-6F;
            TopP = 0.8F;
            Seed = 42;
            Language = "";
            Itn = 0;
            Hotwords = "";
        }

        [MarshalAs(UnmanagedType.LPStr)]
        public string EncoderAdaptor;

        [MarshalAs(UnmanagedType.LPStr)]
        public string LLM;

        [MarshalAs(UnmanagedType.LPStr)]
        public string Embedding;

        [MarshalAs(UnmanagedType.LPStr)]
        public string Tokenizer;

        [MarshalAs(UnmanagedType.LPStr)]
        public string SystemPrompt;

        [MarshalAs(UnmanagedType.LPStr)]
        public string UserPrompt;

        public int MaxNewTokens;
        public float Temperature;
        public float TopP;
        public int Seed;
        [MarshalAs(UnmanagedType.LPStr)]
        public string Language;

        public int Itn;

        [MarshalAs(UnmanagedType.LPStr)]
        public string Hotwords;
    }
}
