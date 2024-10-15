using System.Runtime.InteropServices;

namespace TTS.Struct
{
    [StructLayout(LayoutKind.Sequential)]
    public struct SherpaOnnxOfflineTtsVitsModelConfig
    {
        /// <summary>
        /// 模型
        /// "vits-zh-aishell3/vits-aishell3.onnx"
        /// </summary>
        [MarshalAs(UnmanagedType.LPStr)]
        public string model;
        /// <summary>
        /// 词典文件
        /// "vits-zh-aishell3/lexicon.txt"
        /// </summary>
        [MarshalAs(UnmanagedType.LPStr)]
        public string lexicon;

        [MarshalAs(UnmanagedType.LPStr)]
        public string tokens;

        [MarshalAs(UnmanagedType.LPStr)]
        public string data_dir;

        /// <summary>
        /// VITS模型的noise_scale (float，默认值= 0.667)
        /// </summary>
        public float noise_scale = 0.667f;
        /// <summary>
        /// VITS模型的noise_scale_w (float，默认值= 0.8)
        /// </summary>
        public float noise_scale_w = 0.8f;
        /// <summary>
        /// 演讲的速度。大→慢;小→更快。(float, default = 1)
        /// </summary>
        public float length_scale = 1f;

        [MarshalAs(UnmanagedType.LPStr)]
        public string dict_dir;

        public SherpaOnnxOfflineTtsVitsModelConfig()
        {
            noise_scale = 0.667f;
            noise_scale_w = 0.8f;
            length_scale = 1f;

            model = "vits-zh-aishell3/vits-aishell3.onnx";
            lexicon = "vits-zh-aishell3/lexicon.txt";
            tokens = "vits-zh-aishell3/tokens.txt";
            data_dir = "";
            dict_dir = "";
        }
    }
}
