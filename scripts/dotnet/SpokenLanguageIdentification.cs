/// Copyright (c)  2024.5 by 东风破

using System.Linq;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;
using System;

namespace SherpaOnnx
{
    public class SpokenLanguageIdentification : IDisposable
{
    public SpokenLanguageIdentification(SpokenLanguageIdentificationConfig config)
    {
        IntPtr h = SherpaOnnxCreateSpokenLanguageIdentification(ref config);
        _handle = new HandleRef(this, h);
    }

    public OfflineStream CreateStream()
    {
        IntPtr p = SherpaOnnxSpokenLanguageIdentificationCreateOfflineStream(_handle.Handle);
        return new OfflineStream(p);
    }

    public SpokenLanguageIdentificationResult Compute(OfflineStream stream)
    {
        IntPtr h = SherpaOnnxSpokenLanguageIdentificationCompute(_handle.Handle, stream.Handle);
        SpokenLanguageIdentificationResult result = new SpokenLanguageIdentificationResult(h);
        SherpaOnnxDestroySpokenLanguageIdentificationResult(h);
        return result;
    }

    public void Dispose()
    {
        Cleanup();
        // Prevent the object from being placed on the
        // finalization queue
        System.GC.SuppressFinalize(this);
    }

    ~SpokenLanguageIdentification()
    {
        Cleanup();
    }

    private void Cleanup()
    {
        SherpaOnnxDestroySpokenLanguageIdentification(_handle.Handle);

        // Don't permit the handle to be used again.
        _handle = new HandleRef(this, IntPtr.Zero);
    }

    private HandleRef _handle;

    [DllImport(Dll.Filename)]
    private static extern IntPtr SherpaOnnxCreateSpokenLanguageIdentification(ref SpokenLanguageIdentificationConfig config);

    [DllImport(Dll.Filename)]
    private static extern void SherpaOnnxDestroySpokenLanguageIdentification(IntPtr handle);

    [DllImport(Dll.Filename)]
    private static extern IntPtr SherpaOnnxSpokenLanguageIdentificationCreateOfflineStream(IntPtr handle);

    [DllImport(Dll.Filename)]
    private static extern IntPtr SherpaOnnxSpokenLanguageIdentificationCompute(IntPtr handle, IntPtr stream);

    [DllImport(Dll.Filename)]
    private static extern void SherpaOnnxDestroySpokenLanguageIdentificationResult(IntPtr handle);
}
}