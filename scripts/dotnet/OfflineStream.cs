/// Copyright (c)  2024.5 by 东风破

using System;
using System.Runtime.InteropServices;

namespace SherpaOnnx
{

    public class OfflineStream : IDisposable
    {
        public OfflineStream(IntPtr p)
        {
            _handle = new HandleRef(this, p);
        }

        public void AcceptWaveform(int sampleRate, float[] samples)
        {
            AcceptWaveform(Handle, sampleRate, samples, samples.Length);
        }

        public OfflineRecognizerResult Result
        {
            get
            {
                IntPtr h = GetResult(_handle.Handle);
                OfflineRecognizerResult result = new OfflineRecognizerResult(h);
                DestroyResult(h);
                return result;
            }
        }

        ~OfflineStream()
        {
            Cleanup();
        }

        public void Dispose()
        {
            Cleanup();
            // Prevent the object from being placed on the
            // finalization queue
            System.GC.SuppressFinalize(this);
        }

        private void Cleanup()
        {
            SherpaOnnxDestroyOfflineStream(Handle);

            // Don't permit the handle to be used again.
            _handle = new HandleRef(this, IntPtr.Zero);
        }

        private HandleRef _handle;
        public IntPtr Handle => _handle.Handle;

        [DllImport(Dll.Filename)]
        private static extern void SherpaOnnxDestroyOfflineStream(IntPtr handle);

        [DllImport(Dll.Filename, EntryPoint = "SherpaOnnxAcceptWaveformOffline")]
        private static extern void AcceptWaveform(IntPtr handle, int sampleRate, float[] samples, int n);

        [DllImport(Dll.Filename, EntryPoint = "SherpaOnnxGetOfflineStreamResult")]
        private static extern IntPtr GetResult(IntPtr handle);

        [DllImport(Dll.Filename, EntryPoint = "SherpaOnnxDestroyOfflineRecognizerResult")]
        private static extern void DestroyResult(IntPtr handle);
    }

}
