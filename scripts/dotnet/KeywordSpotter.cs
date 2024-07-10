/// Copyright (c)  2024  Xiaomi Corporation (authors: Fangjun Kuang)
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace SherpaOnnx
{
    // please see
    // https://www.mono-project.com/docs/advanced/pinvoke/#gc-safe-pinvoke-code
    // https://www.mono-project.com/docs/advanced/pinvoke/#properly-disposing-of-resources
    public class KeywordSpotter : IDisposable
    {
        public KeywordSpotter(KeywordSpotterConfig config)
        {
            IntPtr h = CreateKeywordSpotter(ref config);
            _handle = new HandleRef(this, h);
        }

        public OnlineStream CreateStream()
        {
            IntPtr p = CreateKeywordStream(_handle.Handle);
            return new OnlineStream(p);
        }

        public OnlineStream CreateStream(string keywords)
        {
            byte[] utf8Bytes = Encoding.UTF8.GetBytes(keywords);
            IntPtr p = CreateKeywordStreamWithKeywords(_handle.Handle, utf8Bytes);
            return new OnlineStream(p);
        }

        /// Return true if the passed stream is ready for decoding.
        public bool IsReady(OnlineStream stream)
        {
            return IsReady(_handle.Handle, stream.Handle) != 0;
        }

        /// You have to ensure that IsReady(stream) returns true before
        /// you call this method
        public void Decode(OnlineStream stream)
        {
            Decode(_handle.Handle, stream.Handle);
        }

        // The caller should ensure all passed streams are ready for decoding.
        public void Decode(IEnumerable<OnlineStream> streams)
        {
            // TargetFramework=net20 does not support System.Linq
            // IntPtr[] ptrs = streams.Select(s => s.Handle).ToArray();
            List<IntPtr> list = new List<IntPtr>();
            foreach (OnlineStream s in streams)
            {
              list.Add(s.Handle);
            }

            IntPtr[] ptrs = list.ToArray();
            Decode(_handle.Handle, ptrs, ptrs.Length);
        }

        public KeywordResult GetResult(OnlineStream stream)
        {
            IntPtr h = GetResult(_handle.Handle, stream.Handle);
            KeywordResult result = new KeywordResult(h);
            DestroyResult(h);
            return result;
        }

        public void Dispose()
        {
            Cleanup();
            // Prevent the object from being placed on the
            // finalization queue
            System.GC.SuppressFinalize(this);
        }

        ~KeywordSpotter()
        {
            Cleanup();
        }

        private void Cleanup()
        {
            DestroyKeywordSpotter(_handle.Handle);

            // Don't permit the handle to be used again.
            _handle = new HandleRef(this, IntPtr.Zero);
        }

        private HandleRef _handle;

        [DllImport(Dll.Filename)]
        private static extern IntPtr CreateKeywordSpotter(ref KeywordSpotterConfig config);

        [DllImport(Dll.Filename)]
        private static extern void DestroyKeywordSpotter(IntPtr handle);

        [DllImport(Dll.Filename)]
        private static extern IntPtr CreateKeywordStream(IntPtr handle);

        [DllImport(Dll.Filename)]
        private static extern IntPtr CreateKeywordStreamWithKeywords(IntPtr handle, [MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.I1)] byte[] utf8Keywords);

        [DllImport(Dll.Filename, EntryPoint = "IsKeywordStreamReady")]
        private static extern int IsReady(IntPtr handle, IntPtr stream);

        [DllImport(Dll.Filename, EntryPoint = "DecodeKeywordStream")]
        private static extern void Decode(IntPtr handle, IntPtr stream);

        [DllImport(Dll.Filename, EntryPoint = "DecodeMultipleKeywordStreams")]
        private static extern void Decode(IntPtr handle, IntPtr[] streams, int n);

        [DllImport(Dll.Filename, EntryPoint = "GetKeywordResult")]
        private static extern IntPtr GetResult(IntPtr handle, IntPtr stream);

        [DllImport(Dll.Filename, EntryPoint = "DestroyKeywordResult")]
        private static extern void DestroyResult(IntPtr result);
    }
}
