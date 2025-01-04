/// Copyright (c)  2024.5 by 东风破

using System.Runtime.InteropServices;

namespace SherpaOnnx
{

    [StructLayout(LayoutKind.Sequential)]
    public struct OfflineTtsModelConfig
    {
        public OfflineTtsModelConfig()
        {
            Vits = new OfflineTtsVitsModelConfig();
            Matcha = new OfflineTtsMatchaModelConfig();
            NumThreads = 1;
            Debug = 0;
            Provider = "cpu";
        }

        public OfflineTtsVitsModelConfig Vits;
        public OfflineTtsMatchaModelConfig Matcha;
        public int NumThreads;
        public int Debug;
        [MarshalAs(UnmanagedType.LPStr)]
        public string Provider;
    }
}
