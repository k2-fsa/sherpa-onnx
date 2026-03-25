/// Copyright (c)  2026  Xiaomi Corporation (authors: Fangjun Kuang)

using System.Runtime.InteropServices;

namespace SherpaOnnx
{
    [StructLayout(LayoutKind.Sequential)]
    public struct OfflineSourceSeparationModelConfig
    {
        public OfflineSourceSeparationModelConfig()
        {
            Spleeter = new OfflineSourceSeparationSpleeterModelConfig();
            Uvr = new OfflineSourceSeparationUvrModelConfig();
            NumThreads = 1;
            Debug = 0;
            Provider = "cpu";
        }

        public OfflineSourceSeparationSpleeterModelConfig Spleeter;

        public OfflineSourceSeparationUvrModelConfig Uvr;

        public int NumThreads;

        public int Debug;

        [MarshalAs(UnmanagedType.LPStr)]
        public string Provider;
    }
}
