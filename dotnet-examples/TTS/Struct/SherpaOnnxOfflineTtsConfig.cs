using System.Runtime.InteropServices;

namespace TTS.Struct
{
    [StructLayout(LayoutKind.Sequential)]
    public struct SherpaOnnxOfflineTtsConfig
    {
        public SherpaOnnxOfflineTtsModelConfig model;

        [MarshalAs(UnmanagedType.LPStr)]
        public string rule_fsts;

        public int max_num_sentences;

        [MarshalAs(UnmanagedType.LPStr)]
        public string rule_fars;
    }
}
