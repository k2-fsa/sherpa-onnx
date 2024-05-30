/// Copyright (c)  2024.5 by 东风破

using System.Linq;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;
using System;

namespace SherpaOnnx
{
    public class OfflineRecognizer : IDisposable
    {
        public OfflineRecognizer(OfflineRecognizerConfig config)
        {
            IntPtr h = CreateOfflineRecognizer(ref config);
            _handle = new HandleRef(this, h);
        }

        public OfflineStream CreateStream()
        {
            IntPtr p = CreateOfflineStream(_handle.Handle);
            return new OfflineStream(p);
        }

        public void Decode(OfflineStream stream)
        {
            Decode(_handle.Handle, stream.Handle);
        }

        // The caller should ensure all passed streams are ready for decoding.
        public void Decode(IEnumerable<OfflineStream> streams)
        {
            IntPtr[] ptrs = streams.Select(s => s.Handle).ToArray();
            Decode(_handle.Handle, ptrs, ptrs.Length);
        }

        public void Dispose()
        {
            Cleanup();
            // Prevent the object from being placed on the
            // finalization queue
            System.GC.SuppressFinalize(this);
        }

        ~OfflineRecognizer()
        {
            Cleanup();
        }

        private void Cleanup()
        {
            DestroyOfflineRecognizer(_handle.Handle);

            // Don't permit the handle to be used again.
            _handle = new HandleRef(this, IntPtr.Zero);
        }

        private HandleRef _handle;

        [DllImport(Dll.Filename)]
        private static extern IntPtr CreateOfflineRecognizer(ref OfflineRecognizerConfig config);

        [DllImport(Dll.Filename)]
        private static extern void DestroyOfflineRecognizer(IntPtr handle);

        [DllImport(Dll.Filename)]
        private static extern IntPtr CreateOfflineStream(IntPtr handle);

        [DllImport(Dll.Filename, EntryPoint = "DecodeOfflineStream")]
        private static extern void Decode(IntPtr handle, IntPtr stream);

        [DllImport(Dll.Filename, EntryPoint = "DecodeMultipleOfflineStreams")]
        private static extern void Decode(IntPtr handle, IntPtr[] streams, int n);
    }
}
