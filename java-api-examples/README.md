0.Introduction
--------------

Java wrapper `com.k2fsa.sherpa.onnx.OnlineRecognizer` for `sherpa-onnx`. Java is a cross-platform language; you can build jni .so lib according to your system, and then use the same java api for all your platform.
now support multiple threads for websocket server

```xml
Depend on:
  Openjdk 1.8
```

---

1.Compile libsherpa-onnx-jni.so
-------------------------------

Compile sherpa-onnx/jni/jni.cc according to your system.
Example for Ubuntu 18.04 LTS, Openjdk 1.8.0_362:

```xml
  git clone https://github.com/k2-fsa/sherpa-onnx
  cd sherpa-onnx
  mkdir build
  cd build
  cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DSHERPA_ONNX_ENABLE_JNI=ON ..
  make -j6
```

---

2.Download asr model files
--------------------------

[click here for more detail](https://k2-fsa.github.io/sherpa/onnx/pretrained_models/index.html)
--------------------------

3.Config model config.cfg
-------------------------
/**change model path in config.cfg according to your env**/
```xml
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

  #websocket server config
  port=8890
  #number of threads pool for network io
  connection_thread_num=16 
  #number of threads pool for stream
  stream_thread_num=16 
  #number of threads pool for decoder 
  decoder_thread_num=16 
  #size of streams for parallel decoding
  parallel_decoder_num=16
  #time(ms) idle for decoder thread when no job
  decoder_time_idle=10
  #time(ms) out for connection data
  deocder_time_out=3000
```

---

4.A simple java example
-----------------------

refer to [java_api_example](https://github.com/k2-fsa/sherpa-onnx/blob/master/java-api-examples/src/DecodeFile.java) for more detail.

```java
    import com.k2fsa.sherpa.onnx.OnlineRecognizer;
    import com.k2fsa.sherpa.onnx.OnlineStream;
    String cfgpath=appdir+"/modelconfig.cfg";
    OnlineRecognizer.setSoPath(soPath);   //set so lib path

    OnlineRecognizer rcgOjb = new OnlineRecognizer();   //create a recognizer
    rcgOjb = new OnlineRecognizer(cfgFile);    //set model config file 
    CreateStream streamObj=rcgOjb.CreateStream();       //create a stream for read wav data
    float[] buffer = rcgOjb.readWavFile(wavfilename); // read data from file
    streamObj.acceptWaveform(buffer); // feed stream with data
    streamObj.inputFinished(); // tell engine you done with all data
    OnlineStream ssObj[] = new OnlineStream[1];
    while (rcgOjb.isReady(streamObj)) { // engine is ready for unprocessed data
                ssObj[0] = streamObj;
                rcgOjb.decodeStreams(ssObj); // decode for multiple stream
                // rcgOjb.DecodeStream(streamObj);   // decode for single stream
            }

    String recText = "simple:" + rcgOjb.getResult(streamObj) + "\n";
    byte[] utf8Data = recText.getBytes(StandardCharsets.UTF_8);
    System.out.println(new String(utf8Data));
    rcgOjb.reSet(streamObj);
    rcgOjb.releaseStream(streamObj); // release stream
    rcgOjb.release(); // release recognizer
```

---

5.Makefile
----------

OS Ubuntu 18.04 LTS
Build package path: /sherpa-onnx/java-api-examples/lib/sherpaonnx.jar

5.1 Build

```bash
    cd sherpa-onnx/java-api-examples
    make all
```

5.2 Run DecodeFile example

```bash
    make runfile
```

5.3 Run DecodeMic example

```bash
    make runmic
```

---

6.WebSocket Server
----------

support multiple threads for websocket server
6.0 Protocol for communication
1) client connect to server
```shell
   ws client -> srv ws address
   ws address example: ws://127.0.0.1:8889/
```
2) client send 16k pcm_s16le binary stream data to server
```shell
   PCM   sampleRate 16000
         single channel
		 sampleSize 16bit
		 little endian
		 type short
```
3) client send "Done" text to server when all data is sent
```shell
	ws_socket.send("Done")
```
4) client will receive json message from server whenever asr engine decoded new text
```shell
   json example: {"text":"甚至出现交易几乎停滞的情况","eof":false"}
``` 
 

6.1 Build

```bash
    cd sherpa-onnx/java-api-examples
    make all
```

6.2 Run srv example

usage: AsrWebsocketServer soPath modelCfgPath

```bash
    make runsrv  /**change path in Makefile according to your env**/
```

6.3 Run multiple threads client example

usage: AsrWebsocketClient soPath srvIp srvPort wavPath numThreads

json result example: {"text":"甚至出现交易几乎停滞的情况","eof":"true"}

```bash
    make runclient  /**change path in Makefile according to your env**/
```

7 runtest
this script will download model, compile codes and run test
```bash
    cd sherpa-onnx/java-api-examples
    runtest.sh
```