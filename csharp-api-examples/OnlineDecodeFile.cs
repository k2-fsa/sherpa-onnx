// See https://aka.ms/new-console-template for more information
// Copyright (c)  2023 by manyeyes
using SherpaOnnx;
/// Please refer to
/// https://k2-fsa.github.io/sherpa/onnx/pretrained_models/index.html
/// to download pre-trained models. That is, you can find encoder-xxx.onnx
/// decoder-xxx.onnx, joiner-xxx.onnx, and tokens.txt for this struct
/// from there.

/// download model eg:
/// (The directory where the application runs)
/// [/path/to]=System.AppDomain.CurrentDomain.BaseDirectory
/// cd /path/to
/// git clone https://huggingface.co/csukuangfj/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20

/// NuGet for sherpa-onnx
/// PM > Install-Package NAudio -version 2.1.0 -Project sherpa-onnx
/// PM > Install-Package SherpaOnnxCsharp -Project sherpa-onnx

// transducer Usage:
/*
 .\SherpaOnnx.Examples.exe `
  --tokens=./all_models/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/tokens.txt `
  --encoder=./all_models/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/encoder-epoch-99-avg-1.onnx `
  --decoder=./all_models/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/decoder-epoch-99-avg-1.onnx `
  --joiner=./all_models/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/joiner-epoch-99-avg-1.onnx `
  --num-threads=2 `
  --decoding-method=modified_beam_search `
  --debug=false `
  ./all_models/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/test_wavs/0.wav
 */

internal class OnlineDecodeFile
{
    static void Main(string[] args)
    {
        string usage = @"
-----------------------------
transducer Usage:
  --tokens=./all_models/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/tokens.txt `
  --encoder=./all_models/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/encoder-epoch-99-avg-1.onnx `
  --decoder=./all_models/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/decoder-epoch-99-avg-1.onnx `
  --joiner=./all_models/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/joiner-epoch-99-avg-1.onnx `
  --num-threads=2 `
  --decoding-method=modified_beam_search `
  --debug=false `
  ./all_models/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/test_wavs/0.wav
-----------------------------
";
        if (args.Length == 0)
        {
            System.Console.WriteLine("Please enter the correct parameters:");
            System.Console.WriteLine(usage);
            System.Text.StringBuilder sb = new System.Text.StringBuilder();
            //args = Console.ReadLine().Split(" ");
            while (true)
            {
                string input = Console.ReadLine();
                sb.AppendLine(input);
                if (Console.ReadKey().Key == ConsoleKey.Enter)
                    break;
            }
            args = sb.ToString().Split("\r\n");
        }
        Console.WriteLine("Started!\n");
        string? applicationBase = System.AppDomain.CurrentDomain.BaseDirectory;
        List<string> wavFiles = new List<string>();
        Dictionary<string, string> argsDict = GetDict(args, applicationBase, ref wavFiles);
        string decoder = argsDict.ContainsKey("decoder") ? Path.Combine(applicationBase, argsDict["decoder"]) : "";
        string encoder = argsDict.ContainsKey("encoder") ? Path.Combine(applicationBase, argsDict["encoder"]) : "";
        string joiner = argsDict.ContainsKey("joiner") ? Path.Combine(applicationBase, argsDict["joiner"]) : "";
        string paraformer = argsDict.ContainsKey("paraformer") ? Path.Combine(applicationBase, argsDict["paraformer"]) : "";
        string nemo_ctc = argsDict.ContainsKey("nemo_ctc") ? Path.Combine(applicationBase, argsDict["nemo_ctc"]) : "";
        string tokens = argsDict.ContainsKey("tokens") ? Path.Combine(applicationBase, argsDict["tokens"]) : "";
        string num_threads = argsDict.ContainsKey("num_threads") ? argsDict["num_threads"] : "";
        string decoding_method = argsDict.ContainsKey("decoding_method") ? argsDict["decoding_method"] : "";
        string debug = argsDict.ContainsKey("debug") ? argsDict["debug"] : "";

        OfflineTransducer offlineTransducer = new OfflineTransducer();
        offlineTransducer.EncoderFilename = encoder;
        offlineTransducer.DecoderFilename = decoder;
        offlineTransducer.JoinerFilename = joiner;

        OfflineParaformer offlineParaformer = new OfflineParaformer();
        offlineParaformer.Model = paraformer;

        OfflineNemoEncDecCtc offlineNemoEncDecCtc = new OfflineNemoEncDecCtc();
        offlineNemoEncDecCtc.Model = nemo_ctc;

        int numThreads = 0;
        int.TryParse(num_threads, out numThreads);
        bool isDebug = false;
        bool.TryParse(debug, out isDebug);

        string decodingMethod = string.IsNullOrEmpty(decoding_method) ? "" : decoding_method;

        if ((string.IsNullOrEmpty(encoder) || string.IsNullOrEmpty(decoder) || string.IsNullOrEmpty(joiner))
            && string.IsNullOrEmpty(paraformer)
            && string.IsNullOrEmpty(nemo_ctc))
        {
            Console.WriteLine("Please specify at least one model");
            Console.WriteLine(usage);
        }
        // batch decode
        TimeSpan total_duration = TimeSpan.Zero;
        TimeSpan start_time = TimeSpan.Zero;
        TimeSpan end_time = TimeSpan.Zero;
        List<OfflineRecognizerResultEntity> results = new List<OfflineRecognizerResultEntity>();
        if (!(string.IsNullOrEmpty(encoder) || string.IsNullOrEmpty(decoder) || string.IsNullOrEmpty(joiner)))
        {
            OnlineTransducer onlineTransducer = new OnlineTransducer();
            onlineTransducer.EncoderFilename = encoder;
            onlineTransducer.DecoderFilename = decoder;
            onlineTransducer.JoinerFilename = joiner;
            //test online
            OnlineRecognizer<OnlineTransducer> onlineRecognizer = new OnlineRecognizer<OnlineTransducer>(
            onlineTransducer,
            tokens,
            num_threads: numThreads,
            debug: isDebug,
            decoding_method: decodingMethod);
            foreach (string wavFile in wavFiles)
            {
                TimeSpan duration = TimeSpan.Zero;
                List<float[]> samplesList = AudioHelper.GetChunkSamplesList(wavFile, ref duration);
                OnlineStream stream = onlineRecognizer.CreateStream();
                start_time = new TimeSpan(DateTime.Now.Ticks);
                for (int i = 0; i < samplesList.Count; i++)
                {
                    onlineRecognizer.AcceptWaveForm(stream, 16000, samplesList[i]);
                    onlineRecognizer.DecodeStream(stream);
                    OnlineRecognizerResultEntity result_on = onlineRecognizer.GetResult(stream);
                    Console.WriteLine(result_on.text);
                }
                total_duration += duration;
            }
            end_time = new TimeSpan(DateTime.Now.Ticks);
        }
        double elapsed_milliseconds = end_time.TotalMilliseconds - start_time.TotalMilliseconds;
        double rtf = elapsed_milliseconds / total_duration.TotalMilliseconds;
        Console.WriteLine("num_threads:{0}", num_threads);
        Console.WriteLine("decoding_method:{0}", decodingMethod);
        Console.WriteLine("elapsed_milliseconds:{0}", elapsed_milliseconds.ToString());
        Console.WriteLine("wave total_duration_milliseconds:{0}", total_duration.TotalMilliseconds.ToString());
        Console.WriteLine("Real time factor (RTF):{0}", rtf.ToString());

        Console.WriteLine("End!");
    }

    static Dictionary<string, string> GetDict(string[] args, string applicationBase, ref List<string> wavFiles)
    {
        Dictionary<string, string> argsDict = new Dictionary<string, string>();
        foreach (string input in args)
        {
            string[] ss = input.Split("=");
            if (ss.Length == 1)
            {
                if (!string.IsNullOrEmpty(ss[0]))
                {
                    wavFiles.Add(Path.Combine(applicationBase, ss[0].Trim(new char[] { '-', '`', ' ' })));
                }
            }
            else
            {
                argsDict.Add(ss[0].Trim(new char[] { '-', '`', ' ' }).Replace("-", "_"), ss[1].Trim(new char[] { '-', '`', ' ' }));
            }
        }
        return argsDict;
    }

}