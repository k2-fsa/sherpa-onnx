/// Copyright (c)  2026  graehl

using System.Runtime.InteropServices;

namespace SherpaOnnx
{

    [StructLayout(LayoutKind.Sequential)]
    public struct OfflineW2vBertCtcModelConfig
    {
        public OfflineW2vBertCtcModelConfig()
        {
            Model = "";
        }
        [MarshalAs(UnmanagedType.LPStr)]
        public string Model;
    }
}
