/// Copyright (c)  2026  Xiaomi Corporation (authors: Fangjun Kuang)

using System.Runtime.InteropServices;

namespace SherpaOnnx
{
    [StructLayout(LayoutKind.Sequential)]
    public struct OfflineSourceSeparationConfig
    {
        public OfflineSourceSeparationConfig()
        {
            Model = new OfflineSourceSeparationModelConfig();
        }

        public OfflineSourceSeparationModelConfig Model;
    }
}
