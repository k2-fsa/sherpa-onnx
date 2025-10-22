/// Copyright (c)  2024.5 by 东风破

using System.Runtime.InteropServices;

namespace SherpaOnnx
{
    [StructLayout(LayoutKind.Sequential)]
    public struct OfflineTtsModelConfig
    {
        public OfflineTtsModelConfig()
        {
            Vits = new OfflineTtsVitsModelConfig();
            Matcha = new OfflineTtsMatchaModelConfig();
            Kokoro = new OfflineTtsKokoroModelConfig();
            Kitten = new OfflineTtsKittenModelConfig();
            ZipVoice = new OfflineTtsZipVoiceModelConfig();
            NumThreads = 1;
            Debug = 0;
            Provider = "cpu";
        }

        public OfflineTtsVitsModelConfig Vits;
        public int NumThreads;
        public int Debug;

        [MarshalAs(UnmanagedType.LPStr)]
        public string Provider;

        public OfflineTtsMatchaModelConfig Matcha;
        public OfflineTtsKokoroModelConfig Kokoro;
        public OfflineTtsKittenModelConfig Kitten;
        public OfflineTtsZipVoiceModelConfig ZipVoice;
    }
}
