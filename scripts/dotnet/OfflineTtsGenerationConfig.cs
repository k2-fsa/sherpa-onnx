/// Copyright (c)  2026  Xiaomi Corporation (authors: Fangjun Kuang)

using System;
using System.Collections;
using System.Runtime.InteropServices;
using System.Text;

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
            PhonemeCodepoints = null;
            Tokens = null;
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
        /// Pre-phonemized input as Unicode codepoints. Each inner array is one
        /// sentence. When set, it is used in place of the text argument;
        /// leave empty to use text with a lexicon. Mutually exclusive with
        /// Tokens.
        /// </summary>
        public int[][] PhonemeCodepoints;

        /// <summary>
        /// Pre-tokenized input, e.g. Chinese pinyin like "zhong1". Each inner
        /// array is one sentence. Mutually exclusive with PhonemeCodepoints.
        /// </summary>
        public string[][] Tokens;

        /// <summary>
        /// Convert to native struct for P/Invoke
        /// </summary>
        internal NativeStruct ToNative(out NativeResources resources)
        {
            resources = new NativeResources();
            NativeStruct native = new NativeStruct();
            native.SilenceScale = SilenceScale;
            native.Speed = Speed;
            native.Sid = Sid;

            // Handle ReferenceAudio
            if (ReferenceAudio != null && ReferenceAudio.Length > 0)
            {
                resources.AudioHandle = GCHandle.Alloc(ReferenceAudio, GCHandleType.Pinned);
                native.ReferenceAudio = resources.AudioHandle.Value.AddrOfPinnedObject();
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
                    else if (kv.Value is bool)
                        val = (bool)kv.Value ? "true" : "false";
                    else
                        val = kv.Value.ToString();

                    json.AppendFormat("{0}:{1}", key, val);
                }

                json.Append("}");
                native.Extra = json.ToString();
            }
            // Flatten PhonemeCodepoints; the C API takes a flat array plus
            // the length of each sentence.
            if (PhonemeCodepoints != null && PhonemeCodepoints.Length > 0)
            {
                int total = 0;
                foreach (int[] sentence in PhonemeCodepoints)
                {
                    if (sentence != null)
                        total += sentence.Length;
                }

                if (total > 0)
                {
                    int[] flat = new int[total];
                    int[] lens = new int[PhonemeCodepoints.Length];
                    int k = 0;
                    for (int i = 0; i < PhonemeCodepoints.Length; i++)
                    {
                        int[] sentence = PhonemeCodepoints[i];
                        if (sentence == null)
                            continue;
                        lens[i] = sentence.Length;
                        Array.Copy(sentence, 0, flat, k, sentence.Length);
                        k += sentence.Length;
                    }

                    resources.CodepointsHandle = GCHandle.Alloc(flat, GCHandleType.Pinned);
                    resources.CodepointsLensHandle = GCHandle.Alloc(lens, GCHandleType.Pinned);
                    native.PhonemeCodepoints = resources.CodepointsHandle.AddrOfPinnedObject();
                    native.PhonemeCodepointsLens = resources.CodepointsLensHandle.AddrOfPinnedObject();
                    native.PhonemeCodepointsNumSentences = PhonemeCodepoints.Length;
                }
            }

            // Flatten Tokens the same way. Each token string becomes a C
            // string; the char** array is pinned.
            if (Tokens != null && Tokens.Length > 0)
            {
                int total = 0;
                foreach (string[] sentence in Tokens)
                {
                    if (sentence != null)
                        total += sentence.Length;
                }

                if (total > 0)
                {
                    IntPtr[] flat = new IntPtr[total];
                    int[] lens = new int[Tokens.Length];
                    int k = 0;
                    for (int i = 0; i < Tokens.Length; i++)
                    {
                        string[] sentence = Tokens[i];
                        if (sentence == null)
                            continue;
                        lens[i] = sentence.Length;
                        for (int j = 0; j < sentence.Length; j++)
                        {
                            IntPtr p = Marshal.StringToHGlobalAnsi(sentence[j] ?? "");
                            resources.TokenStrings.Add(p);
                            flat[k++] = p;
                        }
                    }

                    resources.TokensHandle = GCHandle.Alloc(flat, GCHandleType.Pinned);
                    resources.TokensLensHandle = GCHandle.Alloc(lens, GCHandleType.Pinned);
                    native.Tokens = resources.TokensHandle.AddrOfPinnedObject();
                    native.TokensLens = resources.TokensLensHandle.AddrOfPinnedObject();
                    native.TokensNumSentences = Tokens.Length;
                }
            }

            return native;
        }

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

            public IntPtr PhonemeCodepoints;
            public IntPtr PhonemeCodepointsLens;
            public int PhonemeCodepointsNumSentences;

            public IntPtr Tokens;
            public IntPtr TokensLens;
            public int TokensNumSentences;
        }
    }
}

