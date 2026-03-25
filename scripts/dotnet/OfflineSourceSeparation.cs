/// Copyright (c)  2026  Xiaomi Corporation (authors: Fangjun Kuang)

using System;
using System.Runtime.InteropServices;

namespace SherpaOnnx
{
    public class OfflineSourceSeparation : IDisposable
    {
        public OfflineSourceSeparation(OfflineSourceSeparationConfig config)
        {
            IntPtr h = SherpaOnnxCreateOfflineSourceSeparation(ref config);
            _handle = new HandleRef(this, h);
        }

        public SourceSeparationOutput Process(float[][] channels, int sampleRate)
        {
            int numChannels = channels.Length;
            if (numChannels == 0)
            {
                return new SourceSeparationOutput(IntPtr.Zero);
            }

            int numSamples = channels[0].Length;

            // Allocate unmanaged memory for the pointer array
            IntPtr ptrArray = Marshal.AllocHGlobal(numChannels * IntPtr.Size);

            // Pin each channel's data
            GCHandle[] handles = new GCHandle[numChannels];
            try
            {
                for (int c = 0; c < numChannels; ++c)
                {
                    handles[c] = GCHandle.Alloc(channels[c], GCHandleType.Pinned);
                    Marshal.WriteIntPtr(ptrArray, c * IntPtr.Size, handles[c].AddrOfPinnedObject());
                }

                IntPtr p = SherpaOnnxOfflineSourceSeparationProcess(_handle.Handle, ptrArray, numChannels, numSamples, sampleRate);
                return new SourceSeparationOutput(p);
            }
            finally
            {
                for (int c = 0; c < numChannels; ++c)
                {
                    if (handles[c].IsAllocated)
                    {
                        handles[c].Free();
                    }
                }
                Marshal.FreeHGlobal(ptrArray);
            }
        }

        public int OutputSampleRate
        {
            get
            {
                return SherpaOnnxOfflineSourceSeparationGetOutputSampleRate(_handle.Handle);
            }
        }

        public int NumberOfStems
        {
            get
            {
                return SherpaOnnxOfflineSourceSeparationGetNumberOfStems(_handle.Handle);
            }
        }

        public void Dispose()
        {
            Cleanup();
            System.GC.SuppressFinalize(this);
        }

        ~OfflineSourceSeparation()
        {
            Cleanup();
        }

        private void Cleanup()
        {
            SherpaOnnxDestroyOfflineSourceSeparation(_handle.Handle);
            _handle = new HandleRef(this, IntPtr.Zero);
        }

        private HandleRef _handle;

        // Read a multi-channel wave file (delegates to C API)
        public static MultiChannelWaveData ReadWaveMultiChannel(string filename)
        {
            byte[] utf8Filename = System.Text.Encoding.UTF8.GetBytes(filename);
            byte[] utf8FilenameWithNull = new byte[utf8Filename.Length + 1];
            Array.Copy(utf8Filename, utf8FilenameWithNull, utf8Filename.Length);
            utf8FilenameWithNull[utf8Filename.Length] = 0;

            IntPtr wave = SherpaOnnxReadWaveMultiChannel(utf8FilenameWithNull);
            if (wave == IntPtr.Zero)
            {
                return null;
            }

            return new MultiChannelWaveData(wave);
        }

        [DllImport(Dll.Filename)]
        private static extern IntPtr SherpaOnnxCreateOfflineSourceSeparation(ref OfflineSourceSeparationConfig config);

        [DllImport(Dll.Filename)]
        private static extern void SherpaOnnxDestroyOfflineSourceSeparation(IntPtr handle);

        [DllImport(Dll.Filename)]
        private static extern int SherpaOnnxOfflineSourceSeparationGetOutputSampleRate(IntPtr handle);

        [DllImport(Dll.Filename)]
        private static extern int SherpaOnnxOfflineSourceSeparationGetNumberOfStems(IntPtr handle);

        [DllImport(Dll.Filename)]
        private static extern IntPtr SherpaOnnxOfflineSourceSeparationProcess(IntPtr handle, IntPtr samples, int num_channels, int num_samples, int sample_rate);

        [DllImport(Dll.Filename)]
        private static extern IntPtr SherpaOnnxReadWaveMultiChannel([MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.I1)] byte[] utf8Filename);

        [DllImport(Dll.Filename)]
        private static extern void SherpaOnnxFreeMultiChannelWave(IntPtr wave);
    }

    // Wrapper for SherpaOnnxMultiChannelWave
    public class MultiChannelWaveData : IDisposable
    {
        internal MultiChannelWaveData(IntPtr handle)
        {
            _handle = new HandleRef(this, handle);
        }

        public int NumChannels
        {
            get
            {
                if (_handle.Handle == IntPtr.Zero) return 0;
                Impl impl = (Impl)Marshal.PtrToStructure(_handle.Handle, typeof(Impl));
                return impl.NumChannels;
            }
        }

        public int NumSamples
        {
            get
            {
                if (_handle.Handle == IntPtr.Zero) return 0;
                Impl impl = (Impl)Marshal.PtrToStructure(_handle.Handle, typeof(Impl));
                return impl.NumSamples;
            }
        }

        public int SampleRate
        {
            get
            {
                if (_handle.Handle == IntPtr.Zero) return 0;
                Impl impl = (Impl)Marshal.PtrToStructure(_handle.Handle, typeof(Impl));
                return impl.SampleRate;
            }
        }

        // Return channels as float[][] for passing to Process()
        public float[][] GetChannels()
        {
            if (_handle.Handle == IntPtr.Zero)
            {
                return new float[0][];
            }

            Impl impl = (Impl)Marshal.PtrToStructure(_handle.Handle, typeof(Impl));
            int nc = impl.NumChannels;
            int ns = impl.NumSamples;

            float[][] result = new float[nc][];
            IntPtr[] channelPtrs = new IntPtr[nc];
            Marshal.Copy(impl.Samples, channelPtrs, 0, nc);

            for (int c = 0; c < nc; ++c)
            {
                result[c] = new float[ns];
                Marshal.Copy(channelPtrs[c], result[c], 0, ns);
            }

            return result;
        }

        public void Dispose()
        {
            if (_handle.Handle != IntPtr.Zero)
            {
                SherpaOnnxFreeMultiChannelWave(_handle.Handle);
            }
            _handle = new HandleRef(this, IntPtr.Zero);
        }

        [StructLayout(LayoutKind.Sequential)]
        struct Impl
        {
            public IntPtr Samples; // const float *const *
            public int NumChannels;
            public int NumSamples;
            public int SampleRate;
        }

        private HandleRef _handle;

        [DllImport(Dll.Filename)]
        private static extern void SherpaOnnxFreeMultiChannelWave(IntPtr wave);
    }
}
