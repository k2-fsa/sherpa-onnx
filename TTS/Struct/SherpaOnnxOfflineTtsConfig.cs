using System.Runtime.InteropServices;

namespace TTS.Struct
{
    [StructLayout(LayoutKind.Sequential)]
    public struct SherpaOnnxOfflineTtsConfig
    {
        public SherpaOnnxOfflineTtsModelConfig model;
    }
}
