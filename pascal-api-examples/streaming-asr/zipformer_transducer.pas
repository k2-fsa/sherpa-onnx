program zipformer_transducer;

{$mode objfpc}

uses
  sherpa_onnx,
  SysUtils;

var
  Config: TSherpaOnnxOnlineRecognizerConfig;
  Recognizer: TSherpaOnnxOnlineRecognizer;
begin
  Initialize(Config);
  Config.ModelConfig.Transducer.Encoder := './sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/encoder-epoch-99-avg-1.int8.onnx';
  Config.ModelConfig.Transducer.Decoder := './sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/decoder-epoch-99-avg-1.onnx';
  Config.ModelConfig.Transducer.Joiner := './sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/joiner-epoch-99-avg-1.int8.onnx';
  Config.ModelConfig.Tokens := './sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/tokens.txt';
  Config.ModelConfig.Debug := True;

  WriteLn(Config.ToString);

  Recognizer := TSherpaOnnxOnlineRecognizer.Create(Config);

  FreeAndNil(Recognizer);

end.
