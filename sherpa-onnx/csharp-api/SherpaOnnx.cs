using System.Runtime.InteropServices;
using System.Diagnostics;

namespace SherpaOnnx
{
    /// <summary>
    /// online recognizer package
    /// Copyright (c)  2023 by manyeyes
    /// </summary>
    public class OnlineBase : IDisposable
    {
        public void Dispose()
        {
            Dispose(disposing: true);
            GC.SuppressFinalize(this);
        }
        protected virtual void Dispose(bool disposing)
        {
            if (!disposing)
            {
                if (_onlineRecognizerResult != IntPtr.Zero)
                {
                    SherpaOnnxSharp.DestroyOnlineRecognizerResult(_onlineRecognizerResult);
                    _onlineRecognizerResult = IntPtr.Zero;
                }
                if (_onlineStream.impl != IntPtr.Zero)
                {
                    SherpaOnnxSharp.DestroyOnlineStream(_onlineStream);
                    _onlineStream.impl = IntPtr.Zero;
                }
                if (_onlineRecognizer.impl != IntPtr.Zero)
                {
                    SherpaOnnxSharp.DestroyOnlineRecognizer(_onlineRecognizer);
                    _onlineRecognizer.impl = IntPtr.Zero;
                }
                this._disposed = true;
            }
        }
        ~OnlineBase()
        {
            Dispose(this._disposed);
        }
        internal SherpaOnnxOnlineStream _onlineStream;
        internal IntPtr _onlineRecognizerResult;
        internal SherpaOnnxOnlineRecognizer _onlineRecognizer;
        internal bool _disposed = false;
    }
    public class OnlineStream : OnlineBase
    {
        internal OnlineStream(SherpaOnnxOnlineStream onlineStream)
        {
            this._onlineStream = onlineStream;
        }
        protected override void Dispose(bool disposing)
        {
            if (!disposing)
            {
                SherpaOnnxSharp.DestroyOnlineStream(_onlineStream);
                _onlineStream.impl = IntPtr.Zero;
                this._disposed = true;
                base.Dispose();
            }
        }
    }
    public class OnlineRecognizerResult : OnlineBase
    {
        internal OnlineRecognizerResult(IntPtr onlineRecognizerResult)
        {
            this._onlineRecognizerResult = onlineRecognizerResult;
        }
        protected override void Dispose(bool disposing)
        {
            if (!disposing)
            {
                SherpaOnnxSharp.DestroyOnlineRecognizerResult(_onlineRecognizerResult);
                _onlineRecognizerResult = IntPtr.Zero;
                this._disposed = true;
                base.Dispose(disposing);
            }
        }
    }
    public class OnlineRecognizer<T> : OnlineBase
        where T : class, new()
    {

        public OnlineRecognizer(T t,
            string tokensFilePath, string decoding_method = "greedy_search",
            int sample_rate = 16000, int feature_dim = 80,
            int num_threads = 2, bool debug = false, int max_active_paths = 4,
            int enable_endpoint=0,int rule1_min_trailing_silence=0,
            int rule2_min_trailing_silence=0,int rule3_min_utterance_length=0)
        {
            SherpaOnnxOnlineTransducer transducer = new SherpaOnnxOnlineTransducer();
            SherpaOnnxOnlineModelConfig model_config = new SherpaOnnxOnlineModelConfig();
            if (t is not null && t.GetType() == typeof(OnlineTransducer))
            {
                OnlineTransducer? onlineTransducer = t as OnlineTransducer;
#pragma warning disable CS8602 // 解引用可能出现空引用。
                Trace.Assert(File.Exists(onlineTransducer.DecoderFilename)
                && File.Exists(onlineTransducer.EncoderFilename)
                && File.Exists(onlineTransducer.JoinerFilename), "Please provide a model");
#pragma warning restore CS8602 // 解引用可能出现空引用。
                Trace.Assert(File.Exists(tokensFilePath), "Please provide a tokens");
                Trace.Assert(num_threads > 0, "num_threads must be greater than 0");
                transducer.encoder_filename = onlineTransducer.EncoderFilename;
                transducer.decoder_filename = onlineTransducer.DecoderFilename;
                transducer.joiner_filename = onlineTransducer.JoinerFilename;
            }

            model_config.transducer = transducer;
            model_config.num_threads = num_threads;
            model_config.debug = debug;
            model_config.tokens = tokensFilePath;

            SherpaOnnxFeatureConfig feat_config = new SherpaOnnxFeatureConfig();
            feat_config.sample_rate = sample_rate;
            feat_config.feature_dim = feature_dim;

            SherpaOnnxOnlineRecognizerConfig sherpaOnnxOnlineRecognizerConfig;
            sherpaOnnxOnlineRecognizerConfig.decoding_method = decoding_method;
            sherpaOnnxOnlineRecognizerConfig.feat_config = feat_config;
            sherpaOnnxOnlineRecognizerConfig.model_config = model_config;
            sherpaOnnxOnlineRecognizerConfig.max_active_paths = max_active_paths;
            //endpoint
            sherpaOnnxOnlineRecognizerConfig.enable_endpoint = enable_endpoint;
            sherpaOnnxOnlineRecognizerConfig.rule1_min_trailing_silence = rule1_min_trailing_silence;
            sherpaOnnxOnlineRecognizerConfig.rule2_min_trailing_silence = rule2_min_trailing_silence;
            sherpaOnnxOnlineRecognizerConfig.rule3_min_utterance_length = rule3_min_utterance_length;

            _onlineRecognizer =
                SherpaOnnxSharp.CreateOnlineRecognizer(sherpaOnnxOnlineRecognizerConfig);
        }
        internal OnlineStream CreateOnlineStream()
        {
            SherpaOnnxOnlineStream stream = SherpaOnnxSharp.CreateOnlineStream(_onlineRecognizer);
            return new OnlineStream(stream);
        }
        public void InputFinished(OnlineStream stream)
        {
            SherpaOnnxSharp.InputFinished(stream._onlineStream);
        }
        public List<OnlineStream> CreateStreams(List<float[]> samplesList)
        {
            int batch_size = samplesList.Count;
            List<OnlineStream> streams = new List<OnlineStream>();
            for (int i = 0; i < batch_size; i++)
            {
                OnlineStream stream = CreateOnlineStream();
                AcceptWaveform(stream._onlineStream, 16000, samplesList[i]);
                InputFinished(stream);
                streams.Add(stream);
            }
            return streams;
        }
        public OnlineStream CreateStream()
        {
            OnlineStream stream = CreateOnlineStream();
            return stream;
        }
        internal void AcceptWaveform(SherpaOnnxOnlineStream stream, int sample_rate, float[] samples)
        {
            SherpaOnnxSharp.AcceptOnlineWaveform(stream, sample_rate, samples, samples.Length);
        }
        public void AcceptWaveForm(OnlineStream stream, int sample_rate, float[] samples)
        {
            AcceptWaveform(stream._onlineStream, sample_rate, samples);
        }
        internal IntPtr GetStreamsIntPtr(OnlineStream[] streams)
        {
            int streams_len = streams.Length;
            int size = Marshal.SizeOf(typeof(SherpaOnnxOnlineStream));
            IntPtr streamsIntPtr = Marshal.AllocHGlobal(size * streams_len);
            unsafe
            {
                byte* ptrbds = (byte*)(streamsIntPtr.ToPointer());
                for (int i = 0; i < streams_len; i++, ptrbds += (size))
                {
                    IntPtr streamIntptr = new IntPtr(ptrbds);
                    Marshal.StructureToPtr(streams[i]._onlineStream, streamIntptr, false);
                }

            }
            return streamsIntPtr;
        }
        internal bool IsReady(OnlineStream stream)
        {
            return SherpaOnnxSharp.IsOnlineStreamReady(_onlineRecognizer, stream._onlineStream) != 0;
        }
        public void DecodeMultipleStreams(List<OnlineStream> streams)
        {
            while (true)
            {
                List<OnlineStream> streamList = new List<OnlineStream>();
                foreach (OnlineStream stream in streams)
                {
                    if (IsReady(stream))
                    {
                        streamList.Add(stream);
                    }
                }
                if (streamList.Count == 0)
                {
                    break;
                }
                OnlineStream[] streamsBatch = new OnlineStream[streamList.Count];
                for (int i = 0; i < streamsBatch.Length; i++)
                {
                    streamsBatch[i] = streamList[i];
                }
                streamList.Clear();
                IntPtr streamsIntPtr = GetStreamsIntPtr(streamsBatch);
                SherpaOnnxSharp.DecodeMultipleOnlineStreams(_onlineRecognizer, streamsIntPtr, streamsBatch.Length);
                Marshal.FreeHGlobal(streamsIntPtr);
            }
        }
        public void DecodeStream(OnlineStream stream)
        {
            while (IsReady(stream))
            {
                SherpaOnnxSharp.DecodeOnlineStream(_onlineRecognizer, stream._onlineStream);
            }
        }
        internal OnlineRecognizerResultEntity GetResult(SherpaOnnxOnlineStream stream)
        {
            IntPtr result_ip = SherpaOnnxSharp.GetOnlineStreamResult(_onlineRecognizer, stream);
            OnlineRecognizerResult onlineRecognizerResult = new OnlineRecognizerResult(result_ip);
#pragma warning disable CS8605 // 取消装箱可能为 null 的值。
            SherpaOnnxOnlineRecognizerResult result =
                (SherpaOnnxOnlineRecognizerResult)Marshal.PtrToStructure(
                    onlineRecognizerResult._onlineRecognizerResult, typeof(SherpaOnnxOnlineRecognizerResult));
#pragma warning restore CS8605 // 取消装箱可能为 null 的值。

#pragma warning disable CS8600 // 将 null 字面量或可能为 null 的值转换为非 null 类型。
            string text = Marshal.PtrToStringAnsi(result.text);
#pragma warning restore CS8600 // 将 null 字面量或可能为 null 的值转换为非 null 类型。
            OnlineRecognizerResultEntity onlineRecognizerResultEntity =
                new OnlineRecognizerResultEntity();
            onlineRecognizerResultEntity.text = text;
            onlineRecognizerResultEntity.text_len = result.text_len;

            return onlineRecognizerResultEntity;
        }
        public OnlineRecognizerResultEntity GetResult(OnlineStream stream)
        {
            OnlineRecognizerResultEntity result = GetResult(stream._onlineStream);
            return result;
        }
        public List<OnlineRecognizerResultEntity> GetResults(List<OnlineStream> streams)
        {
            List<OnlineRecognizerResultEntity> results = new List<OnlineRecognizerResultEntity>();
            foreach (OnlineStream stream in streams)
            {
                OnlineRecognizerResultEntity onlineRecognizerResultEntity = GetResult(stream._onlineStream);
                results.Add(onlineRecognizerResultEntity);
            }
            return results;
        }
        protected override void Dispose(bool disposing)
        {
            if (!disposing)
            {
                SherpaOnnxSharp.DestroyOnlineRecognizer(_onlineRecognizer);
                _onlineRecognizer.impl = IntPtr.Zero;
                this._disposed = true;
                base.Dispose();
            }
        }
    }
    public class OfflineBase : IDisposable
    {
        public void Dispose()
        {
            Dispose(disposing: true);
            GC.SuppressFinalize(this);
        }
        protected virtual void Dispose(bool disposing)
        {
            if (!disposing)
            {
                if (_offlineRecognizerResult != IntPtr.Zero)
                {
                    SherpaOnnxSharp.DestroyOfflineRecognizerResult(_offlineRecognizerResult);
                    _offlineRecognizerResult = IntPtr.Zero;
                }
                if (_offlineStream.impl != IntPtr.Zero)
                {
                    SherpaOnnxSharp.DestroyOfflineStream(_offlineStream);
                    _offlineStream.impl = IntPtr.Zero;
                }
                if (_offlineRecognizer.impl != IntPtr.Zero)
                {
                    SherpaOnnxSharp.DestroyOfflineRecognizer(_offlineRecognizer);
                    _offlineRecognizer.impl = IntPtr.Zero;
                }
                this._disposed = true;
            }
        }
        ~OfflineBase()
        {
            Dispose(this._disposed);
        }
        internal SherpaOnnxOfflineStream _offlineStream;
        internal IntPtr _offlineRecognizerResult;
        internal SherpaOnnxOfflineRecognizer _offlineRecognizer;
        internal bool _disposed = false;
    }
    public class OfflineStream : OfflineBase
    {
        internal OfflineStream(SherpaOnnxOfflineStream offlineStream)
        {
            this._offlineStream = offlineStream;
        }

        protected override void Dispose(bool disposing)
        {
            if (!disposing)
            {
                SherpaOnnxSharp.DestroyOfflineStream(_offlineStream);
                _offlineStream.impl = IntPtr.Zero;
                this._disposed = true;
                base.Dispose();
            }
        }
    }
    public class OfflineRecognizerResult : OfflineBase
    {
        internal OfflineRecognizerResult(IntPtr offlineRecognizerResult)
        {
            this._offlineRecognizerResult = offlineRecognizerResult;
        }
        protected override void Dispose(bool disposing)
        {
            if (!disposing)
            {
                SherpaOnnxSharp.DestroyOfflineRecognizerResult(_offlineRecognizerResult);
                _offlineRecognizerResult = IntPtr.Zero;
                this._disposed = true;
                base.Dispose(disposing);
            }
        }
    }
    public class OfflineRecognizer<T> : OfflineBase
        where T : class, new()
    {
        public OfflineRecognizer(T t,
            string tokensFilePath, string decoding_method = "greedy_search",
            int sample_rate = 16000, int feature_dim = 80,
            int num_threads = 2, bool debug = false)
        {
            SherpaOnnxOfflineTransducer transducer = new SherpaOnnxOfflineTransducer();
            SherpaOnnxOfflineParaformer paraformer = new SherpaOnnxOfflineParaformer();
            SherpaOnnxOfflineNemoEncDecCtc nemo_ctc = new SherpaOnnxOfflineNemoEncDecCtc();
            SherpaOnnxOfflineModelConfig model_config = new SherpaOnnxOfflineModelConfig();
            if (t is not null && t.GetType() == typeof(OfflineTransducer))
            {
                OfflineTransducer? offlineTransducer = t as OfflineTransducer;
#pragma warning disable CS8602 // 解引用可能出现空引用。
                Trace.Assert(File.Exists(offlineTransducer.DecoderFilename)
                && File.Exists(offlineTransducer.EncoderFilename)
                && File.Exists(offlineTransducer.JoinerFilename), "Please provide a model");
#pragma warning restore CS8602 // 解引用可能出现空引用。
                Trace.Assert(File.Exists(tokensFilePath), "Please provide a tokens");
                Trace.Assert(num_threads > 0, "num_threads must be greater than 0");
                transducer.encoder_filename = offlineTransducer.EncoderFilename;
                transducer.decoder_filename = offlineTransducer.DecoderFilename;
                transducer.joiner_filename = offlineTransducer.JoinerFilename;
            }
            else if (t is not null && t.GetType() == typeof(OfflineParaformer))
            {
                OfflineParaformer? offlineParaformer = t as OfflineParaformer;
#pragma warning disable CS8602 // 解引用可能出现空引用。
                Trace.Assert(File.Exists(offlineParaformer.Model), "Please provide a model");
#pragma warning restore CS8602 // 解引用可能出现空引用。
                Trace.Assert(File.Exists(tokensFilePath), "Please provide a tokens");
                Trace.Assert(num_threads > 0, "num_threads must be greater than 0");
                paraformer.model = offlineParaformer.Model;
            }
            else if (t is not null && t.GetType() == typeof(OfflineNemoEncDecCtc))
            {
                OfflineNemoEncDecCtc? offlineNemoEncDecCtc = t as OfflineNemoEncDecCtc;
#pragma warning disable CS8602 // 解引用可能出现空引用。
                Trace.Assert(File.Exists(offlineNemoEncDecCtc.Model), "Please provide a model");
#pragma warning restore CS8602 // 解引用可能出现空引用。
                Trace.Assert(File.Exists(tokensFilePath), "Please provide a tokens");
                Trace.Assert(num_threads > 0, "num_threads must be greater than 0");
                nemo_ctc.model = offlineNemoEncDecCtc.Model;
            }

            model_config.transducer = transducer;
            model_config.paraformer = paraformer;
            model_config.nemo_ctc = nemo_ctc;
            model_config.num_threads = num_threads;
            model_config.debug = debug;
            model_config.tokens = tokensFilePath;

            SherpaOnnxFeatureConfig feat_config = new SherpaOnnxFeatureConfig();
            feat_config.sample_rate = sample_rate;
            feat_config.feature_dim = feature_dim;

            SherpaOnnxOfflineRecognizerConfig sherpaOnnxOfflineRecognizerConfig;
            sherpaOnnxOfflineRecognizerConfig.decoding_method = decoding_method;
            sherpaOnnxOfflineRecognizerConfig.feat_config = feat_config;
            sherpaOnnxOfflineRecognizerConfig.model_config = model_config;

            _offlineRecognizer =
                SherpaOnnxSharp.CreateOfflineRecognizer(sherpaOnnxOfflineRecognizerConfig);
        }
        internal OfflineStream CreateOfflineStream()
        {
            SherpaOnnxOfflineStream stream = SherpaOnnxSharp.CreateOfflineStream(_offlineRecognizer);
            return new OfflineStream(stream);
        }
        public OfflineStream[] CreateOfflineStream(List<float[]> samplesList)
        {
            int batch_size = samplesList.Count;
            OfflineStream[] streams = new OfflineStream[batch_size];
            List<string> wavFiles = new List<string>();
            for (int i = 0; i < batch_size; i++)
            {
                OfflineStream stream = CreateOfflineStream();
                AcceptWaveform(stream._offlineStream, 16000, samplesList[i]);
                streams[i] = stream;
            }
            return streams;
        }
        internal void AcceptWaveform(SherpaOnnxOfflineStream stream, int sample_rate, float[] samples)
        {
            SherpaOnnxSharp.AcceptWaveform(stream, sample_rate, samples, samples.Length);
        }
        internal IntPtr GetStreamsIntPtr(OfflineStream[] streams)
        {
            int streams_len = streams.Length;
            int size = Marshal.SizeOf(typeof(SherpaOnnxOfflineStream));
            IntPtr streamsIntPtr = Marshal.AllocHGlobal(size * streams_len);
            unsafe
            {
                byte* ptrbds = (byte*)(streamsIntPtr.ToPointer());
                for (int i = 0; i < streams_len; i++, ptrbds += (size))
                {
                    IntPtr streamIntptr = new IntPtr(ptrbds);
                    Marshal.StructureToPtr(streams[i]._offlineStream, streamIntptr, false);
                }
            }
            return streamsIntPtr;
        }
        public void DecodeMultipleOfflineStreams(OfflineStream[] streams)
        {
            IntPtr streamsIntPtr = GetStreamsIntPtr(streams);
            SherpaOnnxSharp.DecodeMultipleOfflineStreams(_offlineRecognizer, streamsIntPtr, streams.Length);
            Marshal.FreeHGlobal(streamsIntPtr);
        }
        internal OfflineRecognizerResultEntity GetResult(SherpaOnnxOfflineStream stream)
        {
            IntPtr result_ip = SherpaOnnxSharp.GetOfflineStreamResult(stream);
            OfflineRecognizerResult offlineRecognizerResult = new OfflineRecognizerResult(result_ip);
#pragma warning disable CS8605 // 取消装箱可能为 null 的值。
            SherpaOnnxOfflineRecognizerResult result =
                (SherpaOnnxOfflineRecognizerResult)Marshal.PtrToStructure(
                    offlineRecognizerResult._offlineRecognizerResult, typeof(SherpaOnnxOfflineRecognizerResult));
#pragma warning restore CS8605 // 取消装箱可能为 null 的值。

#pragma warning disable CS8600 // 将 null 字面量或可能为 null 的值转换为非 null 类型。
            string text = Marshal.PtrToStringAnsi(result.text);
#pragma warning restore CS8600 // 将 null 字面量或可能为 null 的值转换为非 null 类型。
            OfflineRecognizerResultEntity offlineRecognizerResultEntity =
                new OfflineRecognizerResultEntity();
            offlineRecognizerResultEntity.text = text;
            offlineRecognizerResultEntity.text_len = result.text_len;

            return offlineRecognizerResultEntity;
        }
        public List<OfflineRecognizerResultEntity> GetResults(OfflineStream[] streams)
        {
            List<OfflineRecognizerResultEntity> results = new List<OfflineRecognizerResultEntity>();
            foreach (OfflineStream stream in streams)
            {
                OfflineRecognizerResultEntity offlineRecognizerResultEntity = GetResult(stream._offlineStream);
                results.Add(offlineRecognizerResultEntity);
            }
            return results;
        }
        protected override void Dispose(bool disposing)
        {
            if (!disposing)
            {
                SherpaOnnxSharp.DestroyOfflineRecognizer(_offlineRecognizer);
                _offlineRecognizer.impl = IntPtr.Zero;
                this._disposed = true;
                base.Dispose();
            }
        }
    }
    internal static partial class SherpaOnnxSharp
    {
        private const string dllName = @"SherpaOnnxSharp";

        [DllImport(dllName, EntryPoint = "CreateOfflineRecognizer", CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl)]
        internal static extern SherpaOnnxOfflineRecognizer CreateOfflineRecognizer(SherpaOnnxOfflineRecognizerConfig config);

        [DllImport(dllName, EntryPoint = "CreateOfflineStream", CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl)]
        internal static extern SherpaOnnxOfflineStream CreateOfflineStream(SherpaOnnxOfflineRecognizer offlineRecognizer);

        [DllImport(dllName, EntryPoint = "AcceptWaveform", CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void AcceptWaveform(SherpaOnnxOfflineStream stream, int sample_rate, float[] samples, int samples_size);

        [DllImport(dllName, EntryPoint = "DecodeOfflineStream", CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void DecodeOfflineStream(SherpaOnnxOfflineRecognizer recognizer, SherpaOnnxOfflineStream stream);

        [DllImport(dllName, EntryPoint = "DecodeMultipleOfflineStreams", CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void DecodeMultipleOfflineStreams(SherpaOnnxOfflineRecognizer recognizer, IntPtr
         streams, int n);

        [DllImport(dllName, EntryPoint = "GetOfflineStreamResult", CallingConvention = CallingConvention.Cdecl)]
        internal static extern IntPtr GetOfflineStreamResult(SherpaOnnxOfflineStream stream);

        [DllImport(dllName, EntryPoint = "DestroyOfflineRecognizerResult", CallingConvention = CallingConvention.Cdecl)]
        internal static extern void DestroyOfflineRecognizerResult(IntPtr result);

        [DllImport(dllName, EntryPoint = "DestroyOfflineStream", CallingConvention = CallingConvention.Cdecl)]
        internal static extern void DestroyOfflineStream(SherpaOnnxOfflineStream stream);

        [DllImport(dllName, EntryPoint = "DestroyOfflineRecognizer", CallingConvention = CallingConvention.Cdecl)]
        internal static extern void DestroyOfflineRecognizer(SherpaOnnxOfflineRecognizer offlineRecognizer);

        [DllImport(dllName, EntryPoint = "CreateOnlineRecognizer", CallingConvention = CallingConvention.Cdecl)]
        internal static extern SherpaOnnxOnlineRecognizer CreateOnlineRecognizer(SherpaOnnxOnlineRecognizerConfig config);

        /// Free a pointer returned by CreateOnlineRecognizer()
        ///
        /// @param p A pointer returned by CreateOnlineRecognizer()
        [DllImport(dllName, EntryPoint = "DestroyOnlineRecognizer", CallingConvention = CallingConvention.Cdecl)]
        internal static extern void DestroyOnlineRecognizer(SherpaOnnxOnlineRecognizer recognizer);

        /// Create an online stream for accepting wave samples.
        ///
        /// @param recognizer  A pointer returned by CreateOnlineRecognizer()
        /// @return Return a pointer to an OnlineStream. The user has to invoke
        ///         DestroyOnlineStream() to free it to avoid memory leak.
        [DllImport(dllName, EntryPoint = "CreateOnlineStream", CallingConvention = CallingConvention.Cdecl)]
        internal static extern SherpaOnnxOnlineStream CreateOnlineStream(
            SherpaOnnxOnlineRecognizer recognizer);

        /// Destroy an online stream.
        ///
        /// @param stream A pointer returned by CreateOnlineStream()
        [DllImport(dllName, EntryPoint = "DestroyOnlineStream", CallingConvention = CallingConvention.Cdecl)]
        internal static extern void DestroyOnlineStream(SherpaOnnxOnlineStream stream);

        /// Accept input audio samples and compute the features.
        /// The user has to invoke DecodeOnlineStream() to run the neural network and
        /// decoding.
        ///
        /// @param stream  A pointer returned by CreateOnlineStream().
        /// @param sample_rate  Sample rate of the input samples. If it is different
        ///                     from config.feat_config.sample_rate, we will do
        ///                     resampling inside sherpa-onnx.
        /// @param samples A pointer to a 1-D array containing audio samples.
        ///                The range of samples has to be normalized to [-1, 1].
        /// @param n  Number of elements in the samples array.
        [DllImport(dllName, EntryPoint = "AcceptOnlineWaveform", CallingConvention = CallingConvention.Cdecl)]
        internal static extern void AcceptOnlineWaveform(SherpaOnnxOnlineStream stream, int sample_rate,
            float[] samples, int n);

        /// Return 1 if there are enough number of feature frames for decoding.
        /// Return 0 otherwise.
        ///
        /// @param recognizer  A pointer returned by CreateOnlineRecognizer
        /// @param stream  A pointer returned by CreateOnlineStream
        [DllImport(dllName, EntryPoint = "IsOnlineStreamReady", CallingConvention = CallingConvention.Cdecl)]
        internal static extern int IsOnlineStreamReady(SherpaOnnxOnlineRecognizer recognizer,
                SherpaOnnxOnlineStream stream);

        /// Call this function to run the neural network model and decoding.
        //
        /// Precondition for this function: IsOnlineStreamReady() MUST return 1.
        ///
        /// Usage example:
        ///
        ///  while (IsOnlineStreamReady(recognizer, stream)) {
        ///     DecodeOnlineStream(recognizer, stream);
        ///  }
        ///
        [DllImport(dllName, EntryPoint = "DecodeOnlineStream", CallingConvention = CallingConvention.Cdecl)]
        internal static extern void DecodeOnlineStream(SherpaOnnxOnlineRecognizer recognizer,
            SherpaOnnxOnlineStream stream);

        /// This function is similar to DecodeOnlineStream(). It decodes multiple
        /// OnlineStream in parallel.
        ///
        /// Caution: The caller has to ensure each OnlineStream is ready, i.e.,
        /// IsOnlineStreamReady() for that stream should return 1.
        ///
        /// @param recognizer  A pointer returned by CreateOnlineRecognizer()
        /// @param streams  A pointer array containing pointers returned by
        ///                 CreateOnlineRecognizer()
        /// @param n  Number of elements in the given streams array.
        [DllImport(dllName, EntryPoint = "DecodeMultipleOnlineStreams", CallingConvention = CallingConvention.Cdecl)]
        internal static extern void DecodeMultipleOnlineStreams(SherpaOnnxOnlineRecognizer recognizer,
            IntPtr streams, int n);

        /// Get the decoding results so far for an OnlineStream.
        ///
        /// @param recognizer A pointer returned by CreateOnlineRecognizer().
        /// @param stream A pointer returned by CreateOnlineStream().
        /// @return A pointer containing the result. The user has to invoke
        ///         DestroyOnlineRecognizerResult() to free the returned pointer to
        ///         avoid memory leak.
        [DllImport(dllName, EntryPoint = "GetOnlineStreamResult", CallingConvention = CallingConvention.Cdecl)]
        internal static extern IntPtr GetOnlineStreamResult(
            SherpaOnnxOnlineRecognizer recognizer, SherpaOnnxOnlineStream stream);

        /// Destroy the pointer returned by GetOnlineStreamResult().
        ///
        /// @param r A pointer returned by GetOnlineStreamResult()
        [DllImport(dllName, EntryPoint = "DestroyOnlineRecognizerResult", CallingConvention = CallingConvention.Cdecl)]
        internal static extern void DestroyOnlineRecognizerResult(IntPtr result);

        /// Reset an OnlineStream , which clears the neural network model state
        /// and the state for decoding.
        ///
        /// @param recognizer A pointer returned by CreateOnlineRecognizer().
        /// @param stream A pointer returned by CreateOnlineStream
        [DllImport(dllName, EntryPoint = "Reset", CallingConvention = CallingConvention.Cdecl)]
        internal static extern void Reset(SherpaOnnxOnlineRecognizer recognizer,
            SherpaOnnxOnlineStream stream);

        /// Signal that no more audio samples would be available.
        /// After this call, you cannot call AcceptWaveform() any more.
        ///
        /// @param stream A pointer returned by CreateOnlineStream()
        [DllImport(dllName, EntryPoint = "InputFinished", CallingConvention = CallingConvention.Cdecl)]
        internal static extern void InputFinished(SherpaOnnxOnlineStream stream);

        /// Return 1 if an endpoint has been detected.
        ///
        /// @param recognizer A pointer returned by CreateOnlineRecognizer()
        /// @param stream A pointer returned by CreateOnlineStream()
        /// @return Return 1 if an endpoint is detected. Return 0 otherwise.
        [DllImport(dllName, EntryPoint = "IsEndpoint", CallingConvention = CallingConvention.Cdecl)]
        internal static extern int IsEndpoint(SherpaOnnxOnlineRecognizer recognizer,
            SherpaOnnxOnlineStream stream);
    }
    internal struct SherpaOnnxOfflineTransducer
    {
        public string encoder_filename;
        public string decoder_filename;
        public string joiner_filename;
        public SherpaOnnxOfflineTransducer()
        {
            encoder_filename = "";
            decoder_filename = "";
            joiner_filename = "";
        }
    };
    internal struct SherpaOnnxOfflineParaformer
    {
        public string model;
        public SherpaOnnxOfflineParaformer()
        {
            model = "";
        }
    };
    internal struct SherpaOnnxOfflineNemoEncDecCtc
    {
        public string model;
        public SherpaOnnxOfflineNemoEncDecCtc()
        {
            model = "";
        }
    };
    internal struct SherpaOnnxOfflineModelConfig
    {
        public SherpaOnnxOfflineTransducer transducer;
        public SherpaOnnxOfflineParaformer paraformer;
        public SherpaOnnxOfflineNemoEncDecCtc nemo_ctc;
        public string tokens;
        public int num_threads;
        public bool debug;
    };
    /// It expects 16 kHz 16-bit single channel wave format.
    internal struct SherpaOnnxFeatureConfig
    {
        /// Sample rate of the input data. MUST match the one expected
        /// by the model. For instance, it should be 16000 for models provided
        /// by us.
        public int sample_rate;

        /// Feature dimension of the model.
        /// For instance, it should be 80 for models provided by us.
        public int feature_dim;
    };
    internal struct SherpaOnnxOfflineRecognizerConfig
    {
        public SherpaOnnxFeatureConfig feat_config;
        public SherpaOnnxOfflineModelConfig model_config;

        /// Possible values are: greedy_search, modified_beam_search
        public string decoding_method;

    };
    internal struct SherpaOnnxOfflineRecognizer
    {
        public IntPtr impl;
    };
    [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Ansi, Pack = 1)]
    internal struct SherpaOnnxOfflineStream
    {
        public IntPtr impl;
    };
    internal struct SherpaOnnxOfflineRecognizerResult
    {
        public IntPtr text;
        public int text_len;
    }
    internal struct SherpaOnnxOnlineTransducer
    {
        public string encoder_filename;
        public string decoder_filename;
        public string joiner_filename;
        public SherpaOnnxOnlineTransducer()
        {
            encoder_filename = string.Empty;
            decoder_filename = string.Empty;
            joiner_filename = string.Empty;
        }
    };
    internal struct SherpaOnnxOnlineModelConfig
    {
        public SherpaOnnxOnlineTransducer transducer;
        public string tokens;
        public int num_threads;
        public bool debug;  // true to print debug information of the model
    };
    internal struct SherpaOnnxOnlineRecognizerConfig
    {
        public SherpaOnnxFeatureConfig feat_config;
        public SherpaOnnxOnlineModelConfig model_config;

        /// Possible values are: greedy_search, modified_beam_search
        public string decoding_method;

        /// Used only when decoding_method is modified_beam_search
        /// Example value: 4
        public int max_active_paths;

        /// 0 to disable endpoint detection.
        /// A non-zero value to enable endpoint detection.
        public int enable_endpoint;

        /// An endpoint is detected if trailing silence in seconds is larger than
        /// this value even if nothing has been decoded.
        /// Used only when enable_endpoint is not 0.
        public float rule1_min_trailing_silence;

        /// An endpoint is detected if trailing silence in seconds is larger than
        /// this value after something that is not blank has been decoded.
        /// Used only when enable_endpoint is not 0.
        public float rule2_min_trailing_silence;

        /// An endpoint is detected if the utterance in seconds is larger than
        /// this value.
        /// Used only when enable_endpoint is not 0.
        public float rule3_min_utterance_length;
    };
    internal struct SherpaOnnxOnlineRecognizerResult
    {
        public IntPtr text;
        public int text_len;
        // TODO: Add more fields
    }
    internal struct SherpaOnnxOnlineRecognizer
    {
        public IntPtr impl;
    };
    [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Ansi, Pack = 1)]
    internal struct SherpaOnnxOnlineStream
    {
        public IntPtr impl;
    };
    public class OfflineNemoEncDecCtc
    {
        private string model = string.Empty;
        public string Model { get => model; set => model = value; }
    }
    public class OfflineParaformer
    {
        private string model = string.Empty;
        public string Model { get => model; set => model = value; }
    }
    public class OfflineRecognizerResultEntity
    {
        /// <summary>
        /// recognizer result
        /// </summary>
        public string? text { get; set; }
        /// <summary>
        /// recognizer result length
        /// </summary>
        public int text_len { get; set; }
        /// <summary>
        /// decode tokens
        /// </summary>
        public List<string>? tokens { get; set; }
        /// <summary>
        /// timestamps
        /// </summary>
        public List<float>? timestamps { get; set; }
    }
    public class OfflineTransducer
    {
        private string encoderFilename = string.Empty;
        private string decoderFilename = string.Empty;
        private string joinerFilename = string.Empty;
        public string EncoderFilename { get => encoderFilename; set => encoderFilename = value; }
        public string DecoderFilename { get => decoderFilename; set => decoderFilename = value; }
        public string JoinerFilename { get => joinerFilename; set => joinerFilename = value; }
    }
    public class OnlineEndpoint
    {
        /// 0 to disable endpoint detection.
        /// A non-zero value to enable endpoint detection.
        private int enableEndpoint;

        /// An endpoint is detected if trailing silence in seconds is larger than
        /// this value even if nothing has been decoded.
        /// Used only when enable_endpoint is not 0.
        private float rule1MinTrailingSilence;

        /// An endpoint is detected if trailing silence in seconds is larger than
        /// this value after something that is not blank has been decoded.
        /// Used only when enable_endpoint is not 0.
        private float rule2MinTrailingSilence;

        /// An endpoint is detected if the utterance in seconds is larger than
        /// this value.
        /// Used only when enable_endpoint is not 0.
        private float rule3MinUtteranceLength;

        public int EnableEndpoint { get => enableEndpoint; set => enableEndpoint = value; }
        public float Rule1MinTrailingSilence { get => rule1MinTrailingSilence; set => rule1MinTrailingSilence = value; }
        public float Rule2MinTrailingSilence { get => rule2MinTrailingSilence; set => rule2MinTrailingSilence = value; }
        public float Rule3MinUtteranceLength { get => rule3MinUtteranceLength; set => rule3MinUtteranceLength = value; }
    }
    public class OnlineRecognizerResultEntity
    {
        /// <summary>
        /// recognizer result
        /// </summary>
        public string? text { get; set; }
        /// <summary>
        /// recognizer result length
        /// </summary>
        public int text_len { get; set; }
        /// <summary>
        /// decode tokens
        /// </summary>
        public List<string>? tokens { get; set; }
        /// <summary>
        /// timestamps
        /// </summary>
        public List<float>? timestamps { get; set; }
    }
    public class OnlineTransducer
    {
        private string encoderFilename = string.Empty;
        private string decoderFilename = string.Empty;
        private string joinerFilename = string.Empty;
        public string EncoderFilename { get => encoderFilename; set => encoderFilename = value; }
        public string DecoderFilename { get => decoderFilename; set => decoderFilename = value; }
        public string JoinerFilename { get => joinerFilename; set => joinerFilename = value; }
    }
}