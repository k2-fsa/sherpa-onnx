/// Copyright (c)  2025  Xiaomi Corporation (authors: Fangjun Kuang)

using System.Runtime.InteropServices;

namespace SherpaOnnx
{
    [StructLayout(LayoutKind.Sequential)]
    public struct OfflineSpeechDenoiserConfig
    {
        public OfflineSpeechDenoiserConfig()
        {
            Model = new OfflineSpeechDenoiserModelConfig();
            DpdfNetAttenuationLimitDb = 0.0f;
        }
        public OfflineSpeechDenoiserModelConfig Model;
        public float DpdfNetAttenuationLimitDb;
    }
}
