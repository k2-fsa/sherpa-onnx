/// Copyright (c)  2023  Xiaomi Corporation (authors: Fangjun Kuang)
/// Copyright (c)  2023 by manyeyes
/// Copyright (c)  2024.5 by 东风破
using System;
using System.Runtime.InteropServices;

namespace SherpaOnnx
{

    public class OnlineStream : IDisposable
    {
        public OnlineStream(IntPtr p)
        {
            _handle = new HandleRef(this, p);
        }

        public void AcceptWaveform(int sampleRate, float[] samples)
        {
            SherpaOnnxOnlineStreamAcceptWaveform(Handle, sampleRate, samples, samples.Length);
        }

        public void InputFinished()
        {
            SherpaOnnxOnlineStreamInputFinished(Handle);
        }

        public void SetOption(string key, string value)
        {
            SherpaOnnxOnlineStreamSetOption(Handle, key, value);
        }

        public string GetOption(string key)
        {
            IntPtr p = SherpaOnnxOnlineStreamGetOption(Handle, key);
            return Marshal.PtrToStringAnsi(p) ?? "";
        }

        public bool HasOption(string key)
        {
            return SherpaOnnxOnlineStreamHasOption(Handle, key) == 1;
        }

        ~OnlineStream()
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
            SherpaOnnxDestroyOnlineStream(Handle);

            // Don't permit the handle to be used again.
            _handle = new HandleRef(this, IntPtr.Zero);
        }

        private HandleRef _handle;
        public IntPtr Handle => _handle.Handle;

        [DllImport(Dll.Filename)]
        private static extern void SherpaOnnxDestroyOnlineStream(IntPtr handle);

        [DllImport(Dll.Filename)]
        private static extern void SherpaOnnxOnlineStreamAcceptWaveform(IntPtr handle, int sampleRate, float[] samples, int n);

        [DllImport(Dll.Filename)]
        private static extern void SherpaOnnxOnlineStreamInputFinished(IntPtr handle);

        [DllImport(Dll.Filename)]
        private static extern void SherpaOnnxOnlineStreamSetOption(IntPtr handle, [MarshalAs(UnmanagedType.LPStr)] string key, [MarshalAs(UnmanagedType.LPStr)] string value);

        [DllImport(Dll.Filename)]
        private static extern IntPtr SherpaOnnxOnlineStreamGetOption(IntPtr handle, [MarshalAs(UnmanagedType.LPStr)] string key);

        [DllImport(Dll.Filename)]
        private static extern int SherpaOnnxOnlineStreamHasOption(IntPtr handle, [MarshalAs(UnmanagedType.LPStr)] string key);
    }

}
