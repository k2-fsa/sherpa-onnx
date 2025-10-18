// Contributor: Eitan (https://github.com/EitanWong)

using System.Runtime.InteropServices;

namespace SherpaOnnx
{
    [StructLayout(LayoutKind.Sequential)]
    public struct OfflineTtsZipvoiceModelConfig
    {
        public OfflineTtsZipvoiceModelConfig()
        {
            Tokens = "";
            TextModel = "";
            FlowMatchingModel = "";
            Vocoder = "";
            DataDir = "";
            PinyinDict = "";

            FeatScale = 0.0f;
            TShift = 0.0f;
            TargetRms = 0.0f;
            GuidanceScale = 0.0f;
        }

        [MarshalAs(UnmanagedType.LPStr)]
        public string Tokens;              // const char* tokens

        [MarshalAs(UnmanagedType.LPStr)]
        public string TextModel;           // const char* text_model

        [MarshalAs(UnmanagedType.LPStr)]
        public string FlowMatchingModel;   // const char* flow_matching_model

        [MarshalAs(UnmanagedType.LPStr)]
        public string Vocoder;             // const char* vocoder

        [MarshalAs(UnmanagedType.LPStr)]
        public string DataDir;             // const char* data_dir

        [MarshalAs(UnmanagedType.LPStr)]
        public string PinyinDict;          // const char* pinyin_dict

        public float FeatScale;            // float feat_scale
        public float TShift;               // float t_shift
        public float TargetRms;            // float target_rms
        public float GuidanceScale;        // float guidance_scale
    }
}