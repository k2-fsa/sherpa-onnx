/// Copyright (c)  2026  Xiaomi Corporation (authors: Fangjun Kuang)

using System.Runtime.InteropServices;

namespace SherpaOnnx
{
    /// <summary>
    /// Per-utterance alignment options, following kalpy's GmmAligner
    /// defaults. A value of 0 lets the C API use its default.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct ForcedAlignmentOptions
    {
        public ForcedAlignmentOptions()
        {
            Beam = 0;
            RetryBeam = 0;
            TransitionScale = 0;
            SelfLoopScale = 0;
            AcousticScale = 0;
        }

        /// <summary>Decoding beam of the first attempt (default 10).</summary>
        public float Beam;

        /// <summary>Beam of the retry when the first attempt fails (default 40).</summary>
        public float RetryBeam;

        /// <summary>Scale of the HMM topology transitions (default 1).</summary>
        public float TransitionScale;

        /// <summary>Scale of the self-loop transitions (default 0.1).</summary>
        public float SelfLoopScale;

        /// <summary>Scale of the acoustic log-likelihoods (default 0.1).</summary>
        public float AcousticScale;
    }
}
