/// Copyright (c)  2023  Xiaomi Corporation (authors: Fangjun Kuang)
/// Copyright (c)  2023 by manyeyes
/// Copyright (c)  2024.5 by 东风破

using System.Runtime.InteropServices;

namespace SherpaOnnx
{

    [StructLayout(LayoutKind.Sequential)]
    public struct OnlineRecognizerConfig
    {
        public OnlineRecognizerConfig()
        {
            FeatConfig = new FeatureConfig();
            ModelConfig = new OnlineModelConfig();
            DecodingMethod = "greedy_search";
            MaxActivePaths = 4;
            EnableEndpoint = 0;
            Rule1MinTrailingSilence = 1.2F;
            Rule2MinTrailingSilence = 2.4F;
            Rule3MinUtteranceLength = 20.0F;
            HotwordsFile = "";
            HotwordsScore = 1.5F;
            CtcFstDecoderConfig = new OnlineCtcFstDecoderConfig();
            RuleFsts = "";
            RuleFars = "";
            BlankPenalty = 0.0F;
        }
        public FeatureConfig FeatConfig;
        public OnlineModelConfig ModelConfig;

        [MarshalAs(UnmanagedType.LPStr)]
        public string DecodingMethod;

        /// Used only when decoding_method is modified_beam_search
        /// Example value: 4
        public int MaxActivePaths;

        /// 0 to disable endpoint detection.
        /// A non-zero value to enable endpoint detection.
        public int EnableEndpoint;

        /// An endpoint is detected if trailing silence in seconds is larger than
        /// this value even if nothing has been decoded.
        /// Used only when enable_endpoint is not 0.
        public float Rule1MinTrailingSilence;

        /// An endpoint is detected if trailing silence in seconds is larger than
        /// this value after something that is not blank has been decoded.
        /// Used only when enable_endpoint is not 0.
        public float Rule2MinTrailingSilence;

        /// An endpoint is detected if the utterance in seconds is larger than
        /// this value.
        /// Used only when enable_endpoint is not 0.
        public float Rule3MinUtteranceLength;

        /// Path to the hotwords.
        [MarshalAs(UnmanagedType.LPStr)]
        public string HotwordsFile;

        /// Bonus score for each token in hotwords.
        public float HotwordsScore;

        public OnlineCtcFstDecoderConfig CtcFstDecoderConfig;

        [MarshalAs(UnmanagedType.LPStr)]
        public string RuleFsts;

        [MarshalAs(UnmanagedType.LPStr)]
        public string RuleFars;

        public float BlankPenalty;
    }
}
