/// Copyright (c)  2023  Xiaomi Corporation (authors: Fangjun Kuang)
/// Copyright (c)  2023 by manyeyes
/// Copyright (c)  2024.5 by 东风破

using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System;

namespace SherpaOnnx
{
    /// It expects 16 kHz 16-bit single channel wave format.
    [StructLayout(LayoutKind.Sequential)]
    public struct FeatureConfig
    {
        public FeatureConfig()
        {
            SampleRate = 16000;
            FeatureDim = 80;
        }
        /// Sample rate of the input data. MUST match the one expected
        /// by the model. For instance, it should be 16000 for models provided
        /// by us.
        public int SampleRate;

        /// Feature dimension of the model.
        /// For instance, it should be 80 for models provided by us.
        public int FeatureDim;
    }

}