/// Copyright (c)  2026  Xiaomi Corporation (authors: Fangjun Kuang)

using System.Runtime.InteropServices;

namespace SherpaOnnx
{
    /// <summary>
    /// Configuration of a forced aligner: a model directory, a
    /// pronunciation dictionary and the default alignment options.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct ForcedAlignerConfig
    {
        public ForcedAlignerConfig()
        {
            ModelDir = "";
            Lexicon = "";
            Options = new ForcedAlignmentOptions();
            Debug = 0;
        }

        /// <summary>
        /// Directory of one forced-alignment acoustic model. Required
        /// files: meta.json, final.alimdl, tree, lda.mat; optional:
        /// phones.txt.
        /// </summary>
        [MarshalAs(UnmanagedType.LPStr)]
        public string ModelDir;

        /// <summary>
        /// Pronunciation dictionary, one pronunciation per line.
        /// </summary>
        [MarshalAs(UnmanagedType.LPStr)]
        public string Lexicon;

        public ForcedAlignmentOptions Options;

        /// <summary>1 to print debug information; 0 to disable it.</summary>
        public int Debug;
    }
}
