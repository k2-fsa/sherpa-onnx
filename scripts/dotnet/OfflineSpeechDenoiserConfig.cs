/// Copyright (c)  2025  Xiaomi Corporation (authors: Fangjun Kuang)

using System.Runtime.InteropServices;

namespace SherpaOnnx
{
    [StructLayout(LayoutKind.Sequential)]
    public struct OfflineSpeechDenoiserConfig
    {
        public OfflineSpeechDenoiserConfig()
        {
            Model = new OfflineSpeechDenoiserConfig();
        }
        public OfflineSpeechDenoiserModelConfig Model;
    }
}
