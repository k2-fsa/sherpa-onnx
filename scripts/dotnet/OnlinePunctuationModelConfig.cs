/// Copyright (c)  2026  Xiaomi Corporation (authors: Fangjun Kuang)

using System.Runtime.InteropServices;

namespace SherpaOnnx
{
    [StructLayout(LayoutKind.Sequential)]
    public struct OnlinePunctuationModelConfig
    {
        public OnlinePunctuationModelConfig()
        {
            CnnBiLstm = "";
            BpeVocab = "";
            NumThreads = 1;
            Debug = 0;
            Provider = "cpu";
        }

        [MarshalAs(UnmanagedType.LPStr)]
        public string CnnBiLstm;

        [MarshalAs(UnmanagedType.LPStr)]
        public string BpeVocab;

        public int NumThreads;

        public int Debug;

        [MarshalAs(UnmanagedType.LPStr)]
        public string Provider;
    }
}
