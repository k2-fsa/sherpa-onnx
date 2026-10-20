/// Copyright (c)  2026  Xiaomi Corporation (authors: Fangjun Kuang)

using System.Runtime.InteropServices;

namespace SherpaOnnx
{
    [StructLayout(LayoutKind.Sequential)]
    public struct OfflineCtcFstDecoderConfig
    {
        public OfflineCtcFstDecoderConfig()
        {
            Graph = "";
            MaxActive = 3000;
        }

        [MarshalAs(UnmanagedType.LPStr)]
        public string Graph;

        /// <summary>
        /// Decoder max-active setting. 0 means 3000.
        /// </summary>
        public int MaxActive;
    }
}
