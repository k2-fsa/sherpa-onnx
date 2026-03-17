/// Copyright (c)  2026  Xiaomi Corporation (authors: Fangjun Kuang)

using System;
using System.Runtime.InteropServices;

namespace SherpaOnnx
{
    public class OnlineSpeechDenoiser: IDisposable
    {
        public OnlineSpeechDenoiser(OnlineSpeechDenoiserConfig config)
        {
            IntPtr h = SherpaOnnxCreateOnlineSpeechDenoiser(ref config);
            _handle = new HandleRef(this, h);
        }

        public DenoisedAudio Run(float[] samples, int sampleRate)
        {
            IntPtr p = SherpaOnnxOnlineSpeechDenoiserRun(_handle.Handle, samples, samples.Length, sampleRate);
            return new DenoisedAudio(p);
        }

        public DenoisedAudio Flush()
        {
            IntPtr p = SherpaOnnxOnlineSpeechDenoiserFlush(_handle.Handle);
            return new DenoisedAudio(p);
        }

        public void Reset()
        {
            SherpaOnnxOnlineSpeechDenoiserReset(_handle.Handle);
        }

        public void Dispose()
        {
            Cleanup();
            System.GC.SuppressFinalize(this);
        }

        ~OnlineSpeechDenoiser()
        {
            Cleanup();
        }

        private void Cleanup()
        {
            SherpaOnnxDestroyOnlineSpeechDenoiser(_handle.Handle);
            _handle = new HandleRef(this, IntPtr.Zero);
        }

        private HandleRef _handle;

        public int SampleRate => SherpaOnnxOnlineSpeechDenoiserGetSampleRate(_handle.Handle);

        public int FrameShiftInSamples =>
            SherpaOnnxOnlineSpeechDenoiserGetFrameShiftInSamples(_handle.Handle);

        [DllImport(Dll.Filename)]
        private static extern IntPtr SherpaOnnxCreateOnlineSpeechDenoiser(ref OnlineSpeechDenoiserConfig config);

        [DllImport(Dll.Filename)]
        private static extern void SherpaOnnxDestroyOnlineSpeechDenoiser(IntPtr handle);

        [DllImport(Dll.Filename)]
        private static extern int SherpaOnnxOnlineSpeechDenoiserGetSampleRate(IntPtr handle);

        [DllImport(Dll.Filename)]
        private static extern int SherpaOnnxOnlineSpeechDenoiserGetFrameShiftInSamples(IntPtr handle);

        [DllImport(Dll.Filename)]
        private static extern IntPtr SherpaOnnxOnlineSpeechDenoiserRun(IntPtr handle, float[] samples, int n, int sampleRate);

        [DllImport(Dll.Filename)]
        private static extern IntPtr SherpaOnnxOnlineSpeechDenoiserFlush(IntPtr handle);

        [DllImport(Dll.Filename)]
        private static extern void SherpaOnnxOnlineSpeechDenoiserReset(IntPtr handle);
    }
}
