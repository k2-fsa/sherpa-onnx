/// Copyright (c)  2026  Xiaomi Corporation (authors: Fangjun Kuang)

using System;
using System.Runtime.InteropServices;
using System.Text;

namespace SherpaOnnx
{
    /// <summary>
    /// Forced aligner: given a wave and its transcript, it computes the
    /// start/end time of every phone and word. Tiers[0] of the result is
    /// the "phones" tier, Tiers[1] the "words" tier.
    /// </summary>
    public class ForcedAligner : IDisposable
    {
        public ForcedAligner(ForcedAlignerConfig config)
        {
            IntPtr h = SherpaOnnxCreateForcedAligner(ref config);
            if (h == IntPtr.Zero)
            {
                throw new ArgumentException(
                    "Failed to create the forced aligner. Please see stderr " +
                    "for the reason; e.g. a missing model file.");
            }
            _handle = new HandleRef(this, h);
        }

        /// <summary>
        /// Create an utterance stream. Accept a waveform with
        /// OfflineStream.AcceptWaveform (float32 samples in [-1, 1]), then
        /// pass the stream to Compute.
        /// </summary>
        public OfflineStream CreateStream()
        {
            IntPtr p = SherpaOnnxForcedAlignerCreateStream(_handle.Handle);
            return new OfflineStream(p);
        }

        /// <summary>
        /// Compute the alignment of one utterance.
        ///
        /// The result is empty (no tiers) with a warning on stderr when the
        /// stream is null, the text is empty, the stream has no features,
        /// every word is out-of-vocabulary, or word boundaries cannot be
        /// recovered. An interval has an empty text for a silence gap in the
        /// "words" tier.
        /// </summary>
        public ForcedAlignmentResult Compute(OfflineStream stream, string text)
        {
            byte[] utf8Bytes = Encoding.UTF8.GetBytes(text);
            byte[] utf8BytesWithNull = new byte[utf8Bytes.Length + 1]; // +1 for null terminator
            Array.Copy(utf8Bytes, utf8BytesWithNull, utf8Bytes.Length);
            utf8BytesWithNull[utf8Bytes.Length] = 0; // Null terminator

            IntPtr p = SherpaOnnxForcedAlignerCompute(
                _handle.Handle, stream.Handle, utf8BytesWithNull);
            if (p == IntPtr.Zero)
            {
                return new ForcedAlignmentResult(IntPtr.Zero);
            }

            try
            {
                return new ForcedAlignmentResult(p);
            }
            finally
            {
                SherpaOnnxDestroyForcedAlignmentResult(p);
            }
        }

        public void Dispose()
        {
            Cleanup();
            // Prevent the object from being placed on the
            // finalization queue
            System.GC.SuppressFinalize(this);
        }

        ~ForcedAligner()
        {
            Cleanup();
        }

        private void Cleanup()
        {
            SherpaOnnxDestroyForcedAligner(_handle.Handle);

            // Don't permit the handle to be used again.
            _handle = new HandleRef(this, IntPtr.Zero);
        }

        private HandleRef _handle;


        [DllImport(Dll.Filename, CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr SherpaOnnxCreateForcedAligner(
            ref ForcedAlignerConfig config);

        [DllImport(Dll.Filename, CallingConvention = CallingConvention.Cdecl)]
        private static extern void SherpaOnnxDestroyForcedAligner(IntPtr handle);

        [DllImport(Dll.Filename, CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr SherpaOnnxForcedAlignerCreateStream(
            IntPtr handle);

        [DllImport(Dll.Filename, CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr SherpaOnnxForcedAlignerCompute(
            IntPtr handle, IntPtr stream,
            [MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.I1)]
                byte[] utf8Text);

        [DllImport(Dll.Filename, CallingConvention = CallingConvention.Cdecl)]
        private static extern void SherpaOnnxDestroyForcedAlignmentResult(
            IntPtr p);
    }
}
