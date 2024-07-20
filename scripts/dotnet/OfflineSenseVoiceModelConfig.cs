/// Copyright (c)  2024  Xiaomi Corporation (authors: Fangjun Kuang)

using System.Runtime.InteropServices;

namespace SherpaOnnx
{
    [StructLayout(LayoutKind.Sequential)]
    public struct OfflineSenseVoiceModelConfig
    {
        public OfflineSenseVoiceModelConfig()
        {
            Model = "";
            Language = "";
            UseInverseTextNormalization = 0;
        }
        [MarshalAs(UnmanagedType.LPStr)]
        public string Model;

        [MarshalAs(UnmanagedType.LPStr)]
        public string Language;

        public int UseInverseTextNormalization;
    }
}
