/// Copyright (c)  2024  Xiaomi Corporation

using System.Runtime.InteropServices;

namespace SherpaOnnx
{

    [StructLayout(LayoutKind.Sequential)]
    public struct OfflineSpeakerSegmentationPyannoteModelConfig
    {
        public OfflineSpeakerSegmentationPyannoteModelConfig()
        {
            Model = "";
            WindowShiftRatio = 0.1f;
        }

        [MarshalAs(UnmanagedType.LPStr)]
        public string Model;
        public float WindowShiftRatio;
    }
}
