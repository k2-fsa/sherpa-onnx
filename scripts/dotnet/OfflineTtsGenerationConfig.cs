/// Copyright (c)  2026  Xiaomi Corporation (authors: Fangjun Kuang)

using System;
using System.Collections;
using System.Runtime.InteropServices;
using System.Text;

#if !NET20
using System.Web.Script.Serialization;
#endif

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
        /// Extra attributes stored as key/value pairs
        /// </summary>
        public Hashtable Extra;

        /// <summary>
        /// Convert to native struct for P/Invoke
        /// </summary>
        internal NativeStruct ToNative(out GCHandle? audioHandle)
        {
            NativeStruct native = new NativeStruct();
            native.SilenceScale = SilenceScale;
            native.Speed = Speed;
            native.Sid = Sid;

            // Handle ReferenceAudio
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

            // Handle Extra JSON
#if NET20
            native.Extra = "{}";
            if (Extra != null && Extra.Count > 0)
            {
                StringBuilder json = new StringBuilder();
                json.Append("{");
                bool first = true;

                foreach (DictionaryEntry kv in Extra)
                {
                    if (!first) json.Append(",");
                    first = false;

                    string key = JsonEscape(kv.Key.ToString());
                    string val;

                    if (kv.Value is string)
                        val = JsonEscape((string)kv.Value);
                    else if (kv.Value is float || kv.Value is double)
                        val = ((IFormattable)kv.Value).ToString(null, System.Globalization.CultureInfo.InvariantCulture);
                    else
                        val = kv.Value.ToString();

                    json.AppendFormat("{0}:{1}", key, val);
                }

                json.Append("}");
                native.Extra = json.ToString();
            }
#else
            if (Extra != null && Extra.Count > 0)
            {
                var serializer = new JavaScriptSerializer();
                native.Extra = serializer.Serialize(Extra);
            }
            else
            {
                native.Extra = "{}";
            }
#endif

            return native;
        }

#if NET20
        /// <summary>
        /// Escapes a string for JSON (for .NET 2.0)
        /// </summary>
        private static string JsonEscape(string s)
        {
            if (s == null) return "\"\"";

            StringBuilder sb = new StringBuilder();
            sb.Append('"');
            foreach (char c in s)
            {
                switch (c)
                {
                    case '"': sb.Append("\\\""); break;
                    case '\\': sb.Append("\\\\"); break;
                    case '\b': sb.Append("\\b"); break;
                    case '\f': sb.Append("\\f"); break;
                    case '\n': sb.Append("\\n"); break;
                    case '\r': sb.Append("\\r"); break;
                    case '\t': sb.Append("\\t"); break;
                    default:
                        if (c < 32 || c > 126)
                            sb.AppendFormat("\\u{0:X4}", (int)c);
                        else
                            sb.Append(c);
                        break;
                }
            }
            sb.Append('"');
            return sb.ToString();
        }
#endif

        [StructLayout(LayoutKind.Sequential)]
        internal struct NativeStruct
        {
            public float SilenceScale;
            public float Speed;
            public int Sid;

            public IntPtr ReferenceAudio;
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

