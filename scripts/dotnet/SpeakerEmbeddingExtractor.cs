/// Copyright (c)  2024.5 by 东风破

using System.Linq;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;
using System;

namespace SherpaOnnx
{
    public class SpeakerEmbeddingExtractor : IDisposable
    {
        public SpeakerEmbeddingExtractor(SpeakerEmbeddingExtractorConfig config)
        {
            IntPtr h = SherpaOnnxCreateSpeakerEmbeddingExtractor(ref config);
            _handle = new HandleRef(this, h);
        }

        public OnlineStream CreateStream()
        {
            IntPtr p = SherpaOnnxSpeakerEmbeddingExtractorCreateStream(_handle.Handle);
            return new OnlineStream(p);
        }

        public bool IsReady(OnlineStream stream)
        {
            return SherpaOnnxSpeakerEmbeddingExtractorIsReady(_handle.Handle, stream.Handle) != 0;
        }

        public float[] Compute(OnlineStream stream)
        {
            IntPtr p = SherpaOnnxSpeakerEmbeddingExtractorComputeEmbedding(_handle.Handle, stream.Handle);

            int dim = Dim;
            float[] ans = new float[dim];
            Marshal.Copy(p, ans, 0, dim);

            SherpaOnnxSpeakerEmbeddingExtractorDestroyEmbedding(p);

            return ans;
        }

        public int Dim
        {
            get
            {
                return SherpaOnnxSpeakerEmbeddingExtractorDim(_handle.Handle);
            }
        }

        public void Dispose()
        {
            Cleanup();
            // Prevent the object from being placed on the
            // finalization queue
            System.GC.SuppressFinalize(this);
        }

        ~SpeakerEmbeddingExtractor()
        {
            Cleanup();
        }

        private void Cleanup()
        {
            SherpaOnnxDestroySpeakerEmbeddingExtractor(_handle.Handle);

            // Don't permit the handle to be used again.
            _handle = new HandleRef(this, IntPtr.Zero);
        }

        private HandleRef _handle;

        [DllImport(Dll.Filename)]
        private static extern IntPtr SherpaOnnxCreateSpeakerEmbeddingExtractor(ref SpeakerEmbeddingExtractorConfig config);

        [DllImport(Dll.Filename)]
        private static extern void SherpaOnnxDestroySpeakerEmbeddingExtractor(IntPtr handle);

        [DllImport(Dll.Filename)]
        private static extern int SherpaOnnxSpeakerEmbeddingExtractorDim(IntPtr handle);

        [DllImport(Dll.Filename)]
        private static extern IntPtr SherpaOnnxSpeakerEmbeddingExtractorCreateStream(IntPtr handle);

        [DllImport(Dll.Filename)]
        private static extern int SherpaOnnxSpeakerEmbeddingExtractorIsReady(IntPtr handle, IntPtr stream);

        [DllImport(Dll.Filename)]
        private static extern IntPtr SherpaOnnxSpeakerEmbeddingExtractorComputeEmbedding(IntPtr handle, IntPtr stream);

        [DllImport(Dll.Filename)]
        private static extern void SherpaOnnxSpeakerEmbeddingExtractorDestroyEmbedding(IntPtr p);
    }

}