/// Copyright (c)  2025  Xiaomi Corporation (authors: Fangjun Kuang)

using System.Runtime.InteropServices;

namespace SherpaOnnx
{
    public struct OfflineSpeechDenoiser: IDisposable
    {
        public OfflineSpeechDenoiser(OfflineSpeechDenoiserConfig config)
        {
            IntPtr h = SherpaOnnxCreateOfflineSpeechDenoiser(ref config);
            _handle = new HandleRef(this, h);
        }

        public DenoisedAudio Run(float[] samples, int sampleRate)
        {
            IntPtr p = SherpaOnnxOfflineSpeechDenoiserRun(_handle.Handle, samples, samples.Length, sampleRate);
            return new DenoisedAudio(p);
        }

        public void Dispose()
        {
            Cleanup();
            // Prevent the object from being placed on the
            // finalization queue
            System.GC.SuppressFinalize(this);
        }

        ~OfflineSpeechDenoiser()
        {
            Cleanup();
        }

        private void Cleanup()
        {
            SherpaOnnxDestroyOfflineSpeechDenoiser(_handle.Handle);

            // Don't permit the handle to be used again.
            _handle = new HandleRef(this, IntPtr.Zero);
        }

        private HandleRef _handle;

        public int SampleRate
        {
            get
            {
                return SherpaOnnxOfflineSpeechDenoiserGetSampleRate(_handle.Handle);
            }
        }

        [DllImport(Dll.Filename)]
        private static extern IntPtr SherpaOnnxCreateOfflineSpeechDenoiser(ref OfflineSpeechDenoiserConfig config);

        [DllImport(Dll.Filename)]
        private static extern void SherpaOnnxDestroyOfflineSpeechDenoiser(IntPtr handle);

        [DllImport(Dll.Filename)]
        private static extern int SherpaOnnxOfflineSpeechDenoiserGetSampleRate(IntPtr handle);

        [DllImport(Dll.Filename)]
        private static extern IntPtr SherpaOnnxOfflineSpeechDenoiserRun(IntPtr handle, float[] samples, int n, int sampleRate);
    }
}
