/// Copyright (c)  2026  Xiaomi Corporation (authors: Fangjun Kuang)

using System;
using System.Collections;
using System.Runtime.InteropServices;

namespace SherpaOnnx
{
    public class OfflineTtsGenerationConfig
    {
        public OfflineTtsGenerationConfig()
        {
            SilenceScale = 0.2f;
            Speed = 1.0f;
            Sid = 0;
            ReferenceAudio = null;
            ReferenceSampleRate = 0;
            ReferenceText = "";
            NumSteps = 5;
            Extra = new Hashtable();
        }

        public float SilenceScale;
        public float Speed;
        public int Sid;

        public float[] ReferenceAudio;
        public int ReferenceSampleRate;
        public string ReferenceText;
        public int NumSteps;

        /// <summary>
        /// Extra attributes serialized as JSON manually
        /// </summary>
        public Hashtable Extra;

        internal NativeStruct ToNative(out GCHandle? audioHandle)
        {
            NativeStruct native = new NativeStruct();
            native.SilenceScale = SilenceScale;
            native.Speed = Speed;
            native.Sid = Sid;

            audioHandle = null;
            if (ReferenceAudio != null && ReferenceAudio.Length > 0)
            {
                audioHandle = GCHandle.Alloc(ReferenceAudio, GCHandleType.Pinned);
                native.ReferenceAudio = audioHandle.Value.AddrOfPinnedObject();
                native.ReferenceAudioLen = ReferenceAudio.Length;
            }
            else
            {
                native.ReferenceAudio = IntPtr.Zero;
                native.ReferenceAudioLen = 0;
            }

            native.ReferenceSampleRate = ReferenceSampleRate;
            native.ReferenceText = ReferenceText ?? "";
            native.NumSteps = NumSteps;

            native.Extra = "{}";
            if (Extra != null && Extra.Count > 0)
            {
                string json = "{";
                bool first = true;
                foreach (System.Collections.DictionaryEntry kv in Extra)
                {
                    if (!first) json += ",";
                    string key = kv.Key.ToString();
                    string val = kv.Value is string ? "\"" + kv.Value.ToString() + "\"" : kv.Value.ToString();
                    json += "\"" + key + "\":" + val;
                    first = false;
                }
                json += "}";
                native.Extra = json;
            }

            return native;
        }




        [StructLayout(LayoutKind.Sequential)]
        internal struct NativeStruct
        {
            public float SilenceScale;
            public float Speed;
            public int Sid;

            public IntPtr ReferenceAudio;   // Use IntPtr for dynamic array
            public int ReferenceAudioLen;
            public int ReferenceSampleRate;

            [MarshalAs(UnmanagedType.LPStr)]
            public string ReferenceText;

            public int NumSteps;

            [MarshalAs(UnmanagedType.LPStr)]
            public string Extra;
        }

    }
}

