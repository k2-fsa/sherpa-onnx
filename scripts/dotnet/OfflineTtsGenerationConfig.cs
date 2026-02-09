/// Copyright (c)  2026  Xiaomi Corporation (authors: Fangjun Kuang)

using System;
using System.Collections;
using System.Runtime.InteropServices;

namespace SherpaOnnx
{
    public class OfflineTtsGenerationConfig
    {
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

        public OfflineTtsGenerationConfig()
        {
            SilenceScale = 1.0f;
            Speed = 1.0f;
            Sid = 0;
            ReferenceAudio = null;
            ReferenceSampleRate = 16000;
            ReferenceText = "";
            NumSteps = 0;
            Extra = new Hashtable();
        }

        internal NativeStruct ToNative()
        {
            NativeStruct native = new NativeStruct();
            native.SilenceScale = SilenceScale;
            native.Speed = Speed;
            native.Sid = Sid;
            native.ReferenceAudio = ReferenceAudio;
            native.ReferenceAudioLen = (ReferenceAudio != null) ? ReferenceAudio.Length : 0;
            native.ReferenceSampleRate = ReferenceSampleRate;
            native.ReferenceText = (ReferenceText != null) ? ReferenceText : "";

            native.NumSteps = NumSteps;

            // Simple JSON string generation
            native.Extra = "{}";
            if (Extra != null && Extra.Count > 0)
            {
                string json = "{";
                bool first = true;
                foreach (DictionaryEntry kv in Extra)
                {
                    if (!first) json += ",";
                    string key = kv.Key.ToString();
                    string val;
                    if (kv.Value is string)
                        val = "\"" + kv.Value.ToString() + "\"";
                    else
                        val = kv.Value.ToString();
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

            public float[] ReferenceAudio;
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

