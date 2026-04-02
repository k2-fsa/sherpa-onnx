/// Copyright (c)  2026  Xiaomi Corporation (authors: Fangjun Kuang)

using System.Runtime.InteropServices;

namespace SherpaOnnx
{
    [StructLayout(LayoutKind.Sequential)]
    public struct OnlinePunctuationConfig
    {
        public OnlinePunctuationConfig()
        {
            Model = new OnlinePunctuationModelConfig();
        }

        public OnlinePunctuationModelConfig Model;
    }
}
