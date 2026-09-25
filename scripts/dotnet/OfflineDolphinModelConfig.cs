/// Copyright (c)  2025  Xiaomi Corporation (authors: Fangjun Kuang)

using System.Runtime.InteropServices;

namespace SherpaOnnx
{
    [StructLayout(LayoutKind.Sequential)]
    public struct OfflineDolphinModelConfig
    {
        public OfflineDolphinModelConfig()
        {
            Model = "";
            Encoder = "";
            Decoder = "";
            Language = "";
            Region = "";
        }
        [MarshalAs(UnmanagedType.LPStr)]
        public string Model;

        [MarshalAs(UnmanagedType.LPStr)]
        public string Encoder;

        [MarshalAs(UnmanagedType.LPStr)]
        public string Decoder;

        [MarshalAs(UnmanagedType.LPStr)]
        public string Language;

        [MarshalAs(UnmanagedType.LPStr)]
        public string Region;
    }
}
