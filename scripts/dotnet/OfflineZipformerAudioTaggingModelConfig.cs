/// Copyright (c)  2025  Xiaomi Corporation (authors: Fangjun Kuang)

using System.Runtime.InteropServices;

namespace SherpaOnnx
{

    [StructLayout(LayoutKind.Sequential)]
    public struct OfflineZipformerAudioTaggingModelConfig
    {
        public OfflineZipformerAudioTaggingModelConfig()
        {
            Model = "";
        }
        [MarshalAs(UnmanagedType.LPStr)]
        public string Model;
    }
}
