{ Copyright (c)  2024  Xiaomi Corporation }
program circular_buffer;
{
This file shows how to use the CircularBuffer API of sherpa-onnx
}

{$mode objfpc}
{$ASSERTIONS ON}

uses
  sherpa_onnx;

var
  Buffer: TSherpaOnnxCircularBuffer;
  Samples: TSherpaOnnxSamplesArray;
begin
  {The initial capacity is 5. It will be resized automatically if needed.}
  Buffer := TSherpaOnnxCircularBuffer.Create(5);
  Assert(Buffer.Size = 0);
  Assert(Buffer.Head = 0);
  Buffer.Push([0, 10, 20]);

  {Push() changes Size. Head is not changed.}
  Assert(Buffer.Size = 3);
  Assert(Buffer.Head = 0);

  Samples := Buffer.Get(0, 1);
  Assert(Length(Samples) = 1);
  Assert(Samples[0] = 0);

  { Get() does not change Size or Head}
  Assert(Buffer.Size = 3);
  Assert(Buffer.Head = 0);

  Samples := Buffer.Get(0, 2);
  Assert(Length(Samples) = 2);
  Assert(Samples[0] = 0);
  Assert(Samples[1] = 10);

  { The buffer will be resized since its initial capacity is 5 but we have
    pushed 7 elements into it.

    No data is lost during the resize.
  }
  Buffer.Push([30, 40, 50, 60]);

  Assert(Buffer.Size = 7); {There are now 7 elements}
  Assert(Buffer.Head = 0);

  {Remove the first 4 elements}
  Buffer.Pop(4);

  Assert(Buffer.Size = 3); {There are only 3 elements left}
  Assert(Buffer.Head = 4);

  Samples := Buffer.Get(Buffer.Head, 2);
  Assert(Length(Samples) = 2);
  Assert(Samples[0] = 40);
  Assert(Samples[1] = 50);

  Buffer.Pop(1);

  Assert(Buffer.Size = 2); {There are only 2 elements left}
  Assert(Buffer.Head = 5);

  Samples := Buffer.Get(Buffer.Head, 2);
  Assert(Length(Samples) = 2);
  Assert(Samples[0] = 50);
  Assert(Samples[1] = 60);

  Buffer.Pop(2);
  Assert(Buffer.Size = 0); {There are no elements left}
  Assert(Buffer.Head = 7);

  Buffer.Push([100, 200, 300, 400, 500]);
  Assert(Buffer.Size = 5);
  Assert(Buffer.Head = 7);

  Buffer.Pop(4);
  Assert(Buffer.Size = 1);

  {Head can be larger than the Capacity!
   This is what circular means. It points to Buffer.Head / Capacity.
  }
  Assert(Buffer.Head = 11);
  Buffer.Push([600, 700]);

  Assert(Buffer.Size = 3);
  Assert(Buffer.Head = 11);

  Samples := Buffer.Get(Buffer.Head, 3);
  Assert(Length(Samples) = 3);
  Assert(Samples[0] = 500);
  Assert(Samples[1] = 600);
  Assert(Samples[2] = 700);

  Buffer.Pop(3);
  Assert(Buffer.Size = 0);
  Assert(Buffer.Head = 14);

  Buffer.Reset();

  Assert(Buffer.Size = 0);
  Assert(Buffer.Head = 0);
end.

