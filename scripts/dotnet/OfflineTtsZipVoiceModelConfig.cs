/// Copyright (c)  2025  Xiaomi Corporation (authors: Fangjun Kuang)

using System.Runtime.InteropServices;

namespace SherpaOnnx
{
    [StructLayout(LayoutKind.Sequential)]
    public struct OfflineTtsZipVoiceModelConfig
    {
        public OfflineTtsZipVoiceModelConfig()
        {
            Tokens = "";
            TextModel = "";
            FlowMatchingModel = "";
            Vocoder = "";
            DataDir = "";
            PinyinDict = "";

            FeatScale = 0.1F;
            Tshift = 0.5F;
            TargetRms = 0.1F;
            GuidanceScale = 1.0F;
        }
        [MarshalAs(UnmanagedType.LPStr)]
        public string Tokens;

        [MarshalAs(UnmanagedType.LPStr)]
        public string TextModel;

        [MarshalAs(UnmanagedType.LPStr)]
        public string FlowMatchingModel;

        [MarshalAs(UnmanagedType.LPStr)]
        public string Vocoder;

        [MarshalAs(UnmanagedType.LPStr)]
        public string DataDir;

        [MarshalAs(UnmanagedType.LPStr)]
        public string PinyinDict;

        public float FeatScale;
        public float Tshift;
        public float TargetRms;
        public float GuidanceScale;
    }
}
