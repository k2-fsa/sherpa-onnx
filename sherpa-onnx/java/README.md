
0.Introduction
---
Java wrapper `com.k2fsa.sherpaonnx.rcglib.OnlineRecognizer` for `sherpa-onnx`. Java is a cross-platform language; you can build jni-java .so lib according to your system, and then use the same java api for all your platform.
``` xml
Depend on:
  Gradle 8.0.2 
  Openjdk 1.8
```
---
1.Compile so. lib
---
Compile sherpa-onnx/jni/jni_java.cc according to your system.
Example for Ubuntu 18.04 LTS, Openjdk 1.8.0_362:
``` xml
  git clone https://github.com/k2-fsa/sherpa-onnx
  cd sherpa-onnx
  mkdir build
  cd build
  cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DSHERPA_ONNX_ENABLE_JNI=ON ..
  make -j6
```
---
2.Download asr model files
---
3.Config model config.cfg
---
``` xml
  #set libsherpa-onnx-jni-java.so lib root dir
  solibpath=/sherpa-onnx/build/lib/libsherpa-onnx-jni-java.so  
  #model config  
  sample_rate=16000                  
  feature_dim=80
  rule1_min_trailing_silence=2.4
  rule2_min_trailing_silence=1.2
  rule3_min_utterance_length=20
  encoder=/sherpa-onnx/build/bin/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/encoder-epoch-99-avg-1.onnx
  decoder=/sherpa-onnx/build/bin/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/decoder-epoch-99-avg-1.onnx
  joiner=/sherpa-onnx/build/bin/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/joiner-epoch-99-avg-1.onnx
  tokens=/sherpa-onnx/build/bin/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/tokens.txt
  num_threads=4
  enable_endpoint_detection=false
  decoding_method=greedy_search
  max_active_paths=4
```
---
4.A simple java example
---
refer to com/k2fsa/sherpaonnx/java_api_example for more detial.
``` java
  import com.k2fsa.sherpaonnx.rcglib.OnlineRecognizer;
  import com.k2fsa.sherpaonnx.rcglib.WavFile;
        OnlineRecognizer.setCfgPath("./modelconfig.cfg"); //set cfg file path
        OnlineRecognizer rcgOjb=new OnlineRecognizer();
		WavFile wavFile = WavFile.openWavFile(new File(wavfilename)); //read wav 
		int numFrame= (int) wavFile.getNumFrames(); //get wav size
		float[] buffer=new float[numFrame];
		int framesRead = wavFile.readFrames(buffer, numFrame);
		rcgOjb.acceptWaveform(buffer,16000);    //feed asr engine in sample rate 16000
		rcgOjb.inputFinished();                //when all wav data is feed to engine
		while (rcgOjb.isReady()){rcgOjb.decode();}  //decode for text
		wavFile.close();
		String recText=rcgOjb.getText();      //get the text
        byte[] utf8Data = recText.getBytes(StandardCharsets.UTF_8);
        System.out.printf(new String(utf8Data));
```
---
5.gradle commands
---
5.1 run app example
``` xml
  export GRADLE_OPTS="-Dfile.encoding=utf-8" //for Chinese 
  gradle run
  ```
5.2 build for jar lib
``` xml
  gradle build 
  ```
 
 
 


 
