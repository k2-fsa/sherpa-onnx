using System.Text;
using TTS;
using TTS.Struct;

internal class Program
{
    private static void Main(string[] args)
    {
        SherpaOnnxOfflineTtsConfig sherpaOnnxOfflineTtsConfig = new SherpaOnnxOfflineTtsConfig();
        sherpaOnnxOfflineTtsConfig.model = new SherpaOnnxOfflineTtsModelConfig
        {
            debug = 0,
            num_threads = 4,
            provider = "cpu",
            vits = new SherpaOnnxOfflineTtsVitsModelConfig
            {
                //lexicon = "vits-zh-aishell3/lexicon.txt",
                //model = "vits-zh-aishell3/vits-aishell3.onnx",
                //tokens = "vits-zh-aishell3/tokens.txt",
                model = @"C:\Services\Sherpa\model.onnx",
                lexicon = "",
                tokens = @"C:\Services\Sherpa\tokens.txt",
                data_dir = @"C:\Services\Sherpa\espeak-ng-data",

                noise_scale = 0.667f,
                noise_scale_w = 0.8f,
                length_scale = 1,
            },

        };

        TTSCore i = new TTSCore(sherpaOnnxOfflineTtsConfig);

        Console.InputEncoding = Encoding.Unicode;
        Console.OutputEncoding = Encoding.UTF8;

        while (true)
        {
            var str = Console.ReadLine();
            var audioResult = i.ToSpeech(str, 40, 1f);

            //  audioResult.WriteWAVFile("123.wav");保存本地

            audioResult.Play();

            int lastIndex = -1;
            while (audioResult.PlayProgress <= 1f)
            {
                int index = (int)(audioResult.PlayProgress * (str.Length - 1));
                if (lastIndex != index)
                {
                    Console.Write(str[index]);
                    lastIndex = index;
                }
                Thread.Sleep(100);
            }

            if (++lastIndex < str.Length)
                Console.Write(str[lastIndex]);

            Console.WriteLine();

        }

    }
}
