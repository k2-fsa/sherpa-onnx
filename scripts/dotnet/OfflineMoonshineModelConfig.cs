/// Copyright (c)  2024-2026  Xiaomi Corporation (authors: Fangjun Kuang)

using System.Runtime.InteropServices;

// For Moonshine v1, you need four models:
//  - preprocessor, encoder, cached_decoder, uncached_decoder
//
// For Moonshine v2, you need 2 models:
//  - encoder, merged_decoder
namespace SherpaOnnx
{
    [StructLayout(LayoutKind.Sequential)]
    public struct OfflineMoonshineModelConfig
    {
        public OfflineMoonshineModelConfig()
        {
            Preprocessor = "";
            Encoder = "";
            UncachedDecoder = "";
            CachedDecoder = "";
            MergedDecoder = "";
        }
        [MarshalAs(UnmanagedType.LPStr)]
        public string Preprocessor;

        [MarshalAs(UnmanagedType.LPStr)]
        public string Encoder;

        [MarshalAs(UnmanagedType.LPStr)]
        public string UncachedDecoder;

        [MarshalAs(UnmanagedType.LPStr)]
        public string CachedDecoder;

        [MarshalAs(UnmanagedType.LPStr)]
        public string MergedDecoder;
    }
}
