/// Copyright (c)  2024.5 by 东风破

using System.Runtime.InteropServices;

namespace SherpaOnnx
{
    [StructLayout(LayoutKind.Sequential)]
    public struct OfflineCanaryModelConfig
    {
        public OfflineCanaryModelConfig()
        {
            Encoder = "";
            Decoder = "";
            SrcLang = "";
            TgtLang = "";
            UsePnc = 1;
        }
        [MarshalAs(UnmanagedType.LPStr)]
        public string Encoder;

        [MarshalAs(UnmanagedType.LPStr)]
        public string Decoder;

        [MarshalAs(UnmanagedType.LPStr)]
        public string SrcLang;

        [MarshalAs(UnmanagedType.LPStr)]
        public string TgtLang;

        public int UsePnc;
    }
}
