/// Copyright (c)  2024.5 by 东风破
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace SherpaOnnx
{
    public class SpeakerEmbeddingManager : IDisposable
    {
        public SpeakerEmbeddingManager(int dim)
        {
            IntPtr h = SherpaOnnxCreateSpeakerEmbeddingManager(dim);
            _handle = new HandleRef(this, h);
            this._dim = dim;
        }

        public bool Add(string name, float[] v)
        {
            return SherpaOnnxSpeakerEmbeddingManagerAdd(_handle.Handle, name, v) == 1;
        }

        public bool Add(string name, ICollection<float[]> v_list)
        {
            int n = v_list.Count;
            float[] v = new float[n * _dim];
            int i = 0;
            foreach (var item in v_list)
            {
                item.CopyTo(v, i);
                i += _dim;
            }

            return SherpaOnnxSpeakerEmbeddingManagerAddListFlattened(_handle.Handle, name, v, n) == 1;
        }

        public bool Remove(string name)
        {
            return SherpaOnnxSpeakerEmbeddingManagerRemove(_handle.Handle, name) == 1;
        }

        public string Search(float[] v, float threshold)
        {
            IntPtr p = SherpaOnnxSpeakerEmbeddingManagerSearch(_handle.Handle, v, threshold);

            string s = "";
            int length = 0;

            unsafe
            {
                byte* b = (byte*)p;
                if (b != null)
                {
                    while (*b != 0)
                    {
                        ++b;
                        length += 1;
                    }
                }
            }

            if (length > 0)
            {
                byte[] stringBuffer = new byte[length];
                Marshal.Copy(p, stringBuffer, 0, length);
                s = Encoding.UTF8.GetString(stringBuffer);
            }

            SherpaOnnxSpeakerEmbeddingManagerFreeSearch(p);

            return s;
        }

        public bool Verify(string name, float[] v, float threshold)
        {
            return SherpaOnnxSpeakerEmbeddingManagerVerify(_handle.Handle, name, v, threshold) == 1;
        }

        public bool Contains(string name)
        {
            return SherpaOnnxSpeakerEmbeddingManagerContains(_handle.Handle, name) == 1;
        }

        public string[] GetAllSpeakers()
        {
            if (NumSpeakers == 0)
            {
                return new string[] { };
            }

            IntPtr names = SherpaOnnxSpeakerEmbeddingManagerGetAllSpeakers(_handle.Handle);

            string[] ans = new string[NumSpeakers];

            unsafe
            {
                byte** p = (byte**)names;
                for (int i = 0; i != NumSpeakers; i++)
                {
                    int length = 0;
                    byte* s = p[i];
                    while (*s != 0)
                    {
                        ++s;
                        length += 1;
                    }
                    byte[] stringBuffer = new byte[length];
                    Marshal.Copy((IntPtr)p[i], stringBuffer, 0, length);
                    ans[i] = Encoding.UTF8.GetString(stringBuffer);
                }
            }

            SherpaOnnxSpeakerEmbeddingManagerFreeAllSpeakers(names);

            return ans;
        }

        public void Dispose()
        {
            Cleanup();
            // Prevent the object from being placed on the
            // finalization queue
            System.GC.SuppressFinalize(this);
        }

        ~SpeakerEmbeddingManager()
        {
            Cleanup();
        }

        private void Cleanup()
        {
            SherpaOnnxDestroySpeakerEmbeddingManager(_handle.Handle);

            // Don't permit the handle to be used again.
            _handle = new HandleRef(this, IntPtr.Zero);
        }

        public int NumSpeakers
        {
            get
            {
                return SherpaOnnxSpeakerEmbeddingManagerNumSpeakers(_handle.Handle);
            }
        }

        private HandleRef _handle;
        private int _dim;


        [DllImport(Dll.Filename)]
        private static extern IntPtr SherpaOnnxCreateSpeakerEmbeddingManager(int dim);

        [DllImport(Dll.Filename)]
        private static extern void SherpaOnnxDestroySpeakerEmbeddingManager(IntPtr handle);

        [DllImport(Dll.Filename)]
        private static extern int SherpaOnnxSpeakerEmbeddingManagerAdd(IntPtr handle, [MarshalAs(UnmanagedType.LPStr)] string name, float[] v);

        [DllImport(Dll.Filename)]
        private static extern int SherpaOnnxSpeakerEmbeddingManagerAddListFlattened(IntPtr handle, [MarshalAs(UnmanagedType.LPStr)] string name, float[] v, int n);

        [DllImport(Dll.Filename)]
        private static extern int SherpaOnnxSpeakerEmbeddingManagerRemove(IntPtr handle, [MarshalAs(UnmanagedType.LPStr)] string name);

        [DllImport(Dll.Filename)]
        private static extern IntPtr SherpaOnnxSpeakerEmbeddingManagerSearch(IntPtr handle, float[] v, float threshold);

        [DllImport(Dll.Filename)]
        private static extern void SherpaOnnxSpeakerEmbeddingManagerFreeSearch(IntPtr p);

        [DllImport(Dll.Filename)]
        private static extern int SherpaOnnxSpeakerEmbeddingManagerVerify(IntPtr handle, [MarshalAs(UnmanagedType.LPStr)] string name, float[] v, float threshold);

        [DllImport(Dll.Filename)]
        private static extern int SherpaOnnxSpeakerEmbeddingManagerContains(IntPtr handle, [MarshalAs(UnmanagedType.LPStr)] string name);

        [DllImport(Dll.Filename)]
        private static extern int SherpaOnnxSpeakerEmbeddingManagerNumSpeakers(IntPtr handle);

        [DllImport(Dll.Filename)]
        private static extern IntPtr SherpaOnnxSpeakerEmbeddingManagerGetAllSpeakers(IntPtr handle);

        [DllImport(Dll.Filename)]
        private static extern void SherpaOnnxSpeakerEmbeddingManagerFreeAllSpeakers(IntPtr names);
    }
}
