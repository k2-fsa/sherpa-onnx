/// Copyright (c)  2026  Xiaomi Corporation (authors: Fangjun Kuang)

using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;

namespace SherpaOnnx
{
    /// <summary>
    /// Owns the native/pinned memory created while marshalling
    /// OfflineTtsGenerationConfig to its native form (reference audio, the
    /// flattened phoneme_codepoints/tokens arrays and the per-token C
    /// strings). Dispose after the native call returns.
    /// </summary>
    internal sealed class NativeResources : IDisposable
    {
        internal GCHandle? AudioHandle;
        internal GCHandle CodepointsHandle;
        internal GCHandle CodepointsLensHandle;
        internal GCHandle TokensHandle;
        internal GCHandle TokensLensHandle;
        internal readonly List<IntPtr> TokenStrings = new List<IntPtr>();

        public void Dispose()
        {
            if (AudioHandle.HasValue)
            {
                AudioHandle.Value.Free();
                AudioHandle = null;
            }

            if (CodepointsHandle.IsAllocated)
                CodepointsHandle.Free();

            if (CodepointsLensHandle.IsAllocated)
                CodepointsLensHandle.Free();

            if (TokensHandle.IsAllocated)
                TokensHandle.Free();

            if (TokensLensHandle.IsAllocated)
                TokensLensHandle.Free();

            foreach (IntPtr p in TokenStrings)
            {
                if (p != IntPtr.Zero)
                    Marshal.FreeHGlobal(p);
            }
            TokenStrings.Clear();
        }
    }
}
