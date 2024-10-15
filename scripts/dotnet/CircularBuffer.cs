/// Copyright (c)  2024  Xiaomi Corporation (authors: Fangjun Kuang)

using System;
using System.Runtime.InteropServices;

namespace SherpaOnnx
{
    public class CircularBuffer : IDisposable
    {
        public CircularBuffer(int capacity)
        {
            IntPtr h = SherpaOnnxCreateCircularBuffer(capacity);
            _handle = new HandleRef(this, h);
        }

        public void Push(float[] data)
        {
            SherpaOnnxCircularBufferPush(_handle.Handle, data, data.Length);
        }

        public float[] Get(int startIndex, int n)
        {
            IntPtr p = SherpaOnnxCircularBufferGet(_handle.Handle, startIndex, n);

            float[] ans = new float[n];
            Marshal.Copy(p, ans, 0, n);

            SherpaOnnxCircularBufferFree(p);

            return ans;
        }

        public void Pop(int n)
        {
            SherpaOnnxCircularBufferPop(_handle.Handle, n);
        }

        public int Size
        {
          get
          {
              return SherpaOnnxCircularBufferSize(_handle.Handle);
          }
        }

        public int Head
        {
          get
          {
              return SherpaOnnxCircularBufferHead(_handle.Handle);
          }
        }

        public void Reset()
        {
            SherpaOnnxCircularBufferReset(_handle.Handle);
        }

        public void Dispose()
        {
            Cleanup();
            // Prevent the object from being placed on the
            // finalization queue
            System.GC.SuppressFinalize(this);
        }

        ~CircularBuffer()
        {
            Cleanup();
        }

        private void Cleanup()
        {
            SherpaOnnxDestroyCircularBuffer(_handle.Handle);

            // Don't permit the handle to be used again.
            _handle = new HandleRef(this, IntPtr.Zero);
        }

        private HandleRef _handle;

        [DllImport(Dll.Filename)]
        private static extern IntPtr SherpaOnnxCreateCircularBuffer(int capacity);

        [DllImport(Dll.Filename)]
        private static extern void SherpaOnnxDestroyCircularBuffer(IntPtr handle);

        [DllImport(Dll.Filename)]
        private static extern void SherpaOnnxCircularBufferPush(IntPtr handle, float[] p, int n);

        [DllImport(Dll.Filename)]
        private static extern IntPtr SherpaOnnxCircularBufferGet(IntPtr handle, int startIndex, int n);

        [DllImport(Dll.Filename)]
        private static extern void SherpaOnnxCircularBufferFree(IntPtr p);

        [DllImport(Dll.Filename)]
        private static extern void SherpaOnnxCircularBufferPop(IntPtr handle, int n);

        [DllImport(Dll.Filename)]
        private static extern int SherpaOnnxCircularBufferSize(IntPtr handle);

        [DllImport(Dll.Filename)]
        private static extern int SherpaOnnxCircularBufferHead(IntPtr handle);

        [DllImport(Dll.Filename)]
        private static extern void SherpaOnnxCircularBufferReset(IntPtr handle);
    }
}
