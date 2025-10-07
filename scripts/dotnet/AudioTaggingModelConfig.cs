/// Copyright (c)  2025  Xiaomi Corporation (authors: Fangjun Kuang)

using System.Runtime.InteropServices;

namespace SherpaOnnx
{
    [StructLayout(LayoutKind.Sequential)]
    public struct AudioTaggingModelConfig
    {
        public AudioTaggingModelConfig()
        {
            Zipformer = new OfflineZipformerAudioTaggingModelConfig();

            CED = "";
            NumThreads = 1;
            Debug = 0;
            Provider = "cpu";
        }

        public OfflineZipformerAudioTaggingModelConfig Zipformer;

        [MarshalAs(UnmanagedType.LPStr)]
        public string CED;

        public int NumThreads;

        public int Debug;

        [MarshalAs(UnmanagedType.LPStr)]
        public string Provider;
    }
}
