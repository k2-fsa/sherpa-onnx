/// Copyright (c)  2024  Xiaomi Corporation
using System;
using System.Runtime.InteropServices;
using System.Text;

namespace SherpaOnnx
{
    // IntPtr is actually a `const float*` from C++
    public delegate int OfflineSpeakerDiarizationProgressCallback(int numProcessedChunks, int numTotalChunks, IntPtr arg);

    public class OfflineSpeakerDiarization : IDisposable
    {
        public OfflineSpeakerDiarization(OfflineSpeakerDiarizationConfig config)
        {
            IntPtr h = SherpaOnnxCreateOfflineSpeakerDiarization(ref config);
            _handle = new HandleRef(this, h);
        }

        public OfflineSpeakerDiarizationSegment[] Process(float[] samples)
        {
            IntPtr result = SherpaOnnxOfflineSpeakerDiarizationProcess(_handle.Handle, samples, samples.Length);
            return ProcessImpl(result);
        }

        public OfflineSpeakerDiarizationSegment[] ProcessWithCallback(float[] samples, OfflineSpeakerDiarizationProgressCallback callback, IntPtr arg)
        {
            IntPtr result = SherpaOnnxOfflineSpeakerDiarizationProcessWithCallback(_handle.Handle, samples, samples.Length, callback, arg);
            return ProcessImpl(result);
        }

        private OfflineSpeakerDiarizationSegment[] ProcessImpl(IntPtr result)
        {
            if (result == IntPtr.Zero)
            {
              return new OfflineSpeakerDiarizationSegment[] {};
            }

            int numSegments = SherpaOnnxOfflineSpeakerDiarizationResultGetNumSegments(result);
            IntPtr p = SherpaOnnxOfflineSpeakerDiarizationResultSortByStartTime(result);

            OfflineSpeakerDiarizationSegment[] ans = new OfflineSpeakerDiarizationSegment[numSegments];
            unsafe
            {
              int size = sizeof(float) * 2 + sizeof(int);
              for (int i = 0; i != numSegments; ++i)
              {
                IntPtr t = new IntPtr((byte*)p + i * size);
                ans[i] = new OfflineSpeakerDiarizationSegment(t);

                // The following IntPtr.Add() does not support net20
                // ans[i] = new OfflineSpeakerDiarizationSegment(IntPtr.Add(p, i));
              }
            }


            SherpaOnnxOfflineSpeakerDiarizationDestroySegment(p);
            SherpaOnnxOfflineSpeakerDiarizationDestroyResult(result);

            return ans;

        }

        public void Dispose()
        {
            Cleanup();
            // Prevent the object from being placed on the
            // finalization queue
            System.GC.SuppressFinalize(this);
        }

        ~OfflineSpeakerDiarization()
        {
            Cleanup();
        }

        private void Cleanup()
        {
            SherpaOnnxDestroyOfflineSpeakerDiarization(_handle.Handle);

            // Don't permit the handle to be used again.
            _handle = new HandleRef(this, IntPtr.Zero);
        }

        private HandleRef _handle;

        public int SampleRate
        {
            get
            {
                return SherpaOnnxOfflineSpeakerDiarizationGetSampleRate(_handle.Handle);
            }
        }

        [DllImport(Dll.Filename)]
        private static extern IntPtr SherpaOnnxCreateOfflineSpeakerDiarization(ref OfflineSpeakerDiarizationConfig config);

        [DllImport(Dll.Filename)]
        private static extern void SherpaOnnxDestroyOfflineSpeakerDiarization(IntPtr handle);

        [DllImport(Dll.Filename)]
        private static extern int SherpaOnnxOfflineSpeakerDiarizationGetSampleRate(IntPtr handle);

        [DllImport(Dll.Filename)]
        private static extern int SherpaOnnxOfflineSpeakerDiarizationResultGetNumSegments(IntPtr handle);

        [DllImport(Dll.Filename)]
        private static extern IntPtr SherpaOnnxOfflineSpeakerDiarizationProcess(IntPtr handle, float[] samples, int n);

        [DllImport(Dll.Filename, CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr SherpaOnnxOfflineSpeakerDiarizationProcessWithCallback(IntPtr handle, float[] samples, int n, OfflineSpeakerDiarizationProgressCallback callback, IntPtr arg);

        [DllImport(Dll.Filename)]
        private static extern void SherpaOnnxOfflineSpeakerDiarizationDestroyResult(IntPtr handle);

        [DllImport(Dll.Filename)]
        private static extern IntPtr SherpaOnnxOfflineSpeakerDiarizationResultSortByStartTime(IntPtr handle);

        [DllImport(Dll.Filename)]
        private static extern void SherpaOnnxOfflineSpeakerDiarizationDestroySegment(IntPtr handle);
    }
}

