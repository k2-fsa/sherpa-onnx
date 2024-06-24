/// Copyright (c)  2024.5 by 东风破

using System.Runtime.InteropServices;

namespace SherpaOnnx
{

    [StructLayout(LayoutKind.Sequential)]
    public struct OfflineRecognizerConfig
    {
        public OfflineRecognizerConfig()
        {
            FeatConfig = new FeatureConfig();
            ModelConfig = new OfflineModelConfig();
            LmConfig = new OfflineLMConfig();

            DecodingMethod = "greedy_search";
            MaxActivePaths = 4;
            HotwordsFile = "";
            HotwordsScore = 1.5F;
            RuleFsts = "";
            RuleFars = "";
        }
        public FeatureConfig FeatConfig;
        public OfflineModelConfig ModelConfig;
        public OfflineLMConfig LmConfig;

        [MarshalAs(UnmanagedType.LPStr)]
        public string DecodingMethod;

        public int MaxActivePaths;

        [MarshalAs(UnmanagedType.LPStr)]
        public string HotwordsFile;

        public float HotwordsScore;

        [MarshalAs(UnmanagedType.LPStr)]
        public string RuleFsts;

        [MarshalAs(UnmanagedType.LPStr)]
        public string RuleFars;
    }


}