/// Copyright (c)  2026  Xiaomi Corporation

using System.Runtime.InteropServices;

namespace SherpaOnnx
{
    [StructLayout(LayoutKind.Sequential)]
    public struct OfflineSpeechDenoiserDpdfNetModelConfig
    {
        public OfflineSpeechDenoiserDpdfNetModelConfig()
        {
            Model = "";
            AttenuationLimitDb = 0.0f;
        }

        [MarshalAs(UnmanagedType.LPStr)]
        public string Model;
        public float AttenuationLimitDb;
    }
}
