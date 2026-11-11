/// Copyright (c)  2024  Xiaomi Corporation

using System.Runtime.InteropServices;

namespace SherpaOnnx
{

    [StructLayout(LayoutKind.Sequential)]
    public struct FastClusteringConfig
    {
        public FastClusteringConfig()
        {
            NumClusters = -1;
            Threshold = 0.5F;
            ComputeConfidence = 0;
        }

        public int NumClusters;
        public float Threshold;
        public int ComputeConfidence;
    }
}
