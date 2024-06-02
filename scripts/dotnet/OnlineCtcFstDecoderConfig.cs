/// Copyright (c)  2023  Xiaomi Corporation (authors: Fangjun Kuang)
/// Copyright (c)  2023 by manyeyes
/// Copyright (c)  2024.5 by 东风破

using System.Runtime.InteropServices;

namespace SherpaOnnx
{
    [StructLayout(LayoutKind.Sequential)]
    public struct OnlineCtcFstDecoderConfig
    {
        public OnlineCtcFstDecoderConfig()
        {
            Graph = "";
            MaxActive = 3000;
        }

        [MarshalAs(UnmanagedType.LPStr)]
        public string Graph;

        public int MaxActive;
    }

}