// pascal-api-examples/version/version.pas
//
// Copyright (c)  2026  Xiaomi Corporation

program version;

uses
  sherpa_onnx;

begin
  WriteLn('sherpa-onnx version : ', SherpaOnnxGetVersionStr());
  WriteLn('sherpa-onnx Git SHA1: ', SherpaOnnxGetGitSha1());
  WriteLn('sherpa-onnx Git date: ', SherpaOnnxGetGitDate());
  WriteLn('onnxruntime version : ', SherpaOnnxGetOnnxruntimeVersionStr());
end.
