## 1.12.15

* Exposing online punctuation model support in node-addon-api (#2609)
* Fix building wheels (#2619)
* Export one more Piper Arabic TTS model (#2623)
* fix: hot update language for sencevoice (#2627)
* Add C API and Go API for Zipvoice (#2628)
* Add CI tests for Zipvoice Go API (#2630)
* Remove hardcoded dithering value in NeMo transducer recognizer (#2639)
* Reduce verbose output about reading lexicon for TTS (#2648)
* Add Parakeet TDT model for generating subtitles (#2649)
* Add more Piper TTS models (#2651)
* Add CXX API for audio tagging (#2652)
* Add C# API for audio tagging (#2653)
* Support KWS + RKNN. (#2190)
* Support https://github.com/ASLP-lab/WenetSpeech-Chuan (#2656)
* Fix building for android (#2657)
* fix ios build script (#2645)
* Update kaldi-native-fbank (#2659)
* Add missing python class definitions for builds without TTS support (#2660)
* Remove jieba from kokoro and matcha tts. (#2662)
* add flet_sherpa_onnx in readme (#2663)
* Remove cppjieba (#2664)
* Add phrase matcher to merge words into phrases for TTS. (#2668)
* Limit number of tokens per sentence in MatchaTTS. (#2671)
* Update README to include a ROS2 project using sherpa-onnx (#2672)
* Fix building Flutter APPs (#2673)
* Export Paraformer to RKNN (#2689)
* Update README.md add achatbot-go Projects using sherpa-onnx link (#2691)
* Add CI to export Paraformer to RKNN (#2692)
* Support MatchTTS with English and Chinese (#2695)
* Export Paraformer ASR models from FunASR to Ascend NPU 910B (#2697)
* Update README to include Ascend NPU (#2698)
* Fix WASM (JS) after adding zipvoice. (#2702)
* Export SenseVoice ASR models to Ascend NPU 910B (#2707)
* Fix building for various language bindings after adding zipvoice (#2709)

## 1.12.14

* Fix setting rknn core mask (#2594)
* Add Dart API for spoken language identification (#2596)
* Add CI tests for dart spoken language identifcation example (#2598)
* Provide pre-compiled shepra-onnx libs/binaries for CUDA 12.x + onnxruntime 1.22.0 (#2599)
* Provide pre-compiled whls for cuda 12.x on Linux x64 and Windows x64 (#2601)
* Fix TDT decoding for NeMo TDT transducers (#2606)
* Add a C++ example for simulated streaming ASR (#2607)

## 1.12.13

* Fix initializing symbol table for OnlineRecognizer. (#2590)
* Support RK NPU for SenseVoice non-streaming ASR models (#2589)
* Upload RKNN models for sense-voice (#2592)

## 1.12.12

* Fix building for risc-v (#2549)
* Fix using sherpa-onnx as a cmake sub-project. (#2550)
* Update kaldifst and kaldi-decoder (#2551)
* Support armv8l in Java API (#2556)
* Disable loading libs from jar on Android. (#2557)
* Fix cantonese vits tts (#2558)
* Avoid appending blanks for Cantonese vits tts. (#2559)
* Add hint for loading model files from SD card on Android. (#2564)
* Update README to include https://github.com/Mentra-Community/MentraOS (#2565)
* Export models from https://github.com/voicekit-team/T-one to sherpa-onnx (#2571)
* Add C++ and Python support for T-one streaming Russian ASR models (#2575)
* Add various language bindings for streaming T-one Russian ASR models (#2576)
* Fix the missing online punctuation in android aar (#2577)
* Export KittenTTS mini v0.1 to sherpa-onnx (#2578)
* Upload new sense-voice models (#2580)
* Export ASLP-lab/WSYue-ASR/tree/main/u2pp_conformer_yue to sherpa-onnx (#2582)
* Add various languge bindings for Wenet non-streaming CTC models (#2584)

## 1.12.11

* Add two more Piper tts models (#2525)
* Generate tts samples for MatchaTTS (English). (#2527)
* Fix releasing go packages (#2529)
* Add license info about tts models from OpenVoiceOS (#2530)
* Support BPE models with byte fallback. (#2531)
* Simplify the usage of our non-Android Java API (#2533)
* Fix wasm for kws (#2535)
* Add one more German tts model from OpenVoiceOS. (#2536)
* Fix uploading win32 libs to huggingface (#2537)
* Add Zipvoice (#2487)
* Fix c api (#2545)
* Fix linking (#2546)

## 1.12.10

* Add VOSK streaming Russian ASR models and Kroko streaming German ASR models (#2502)
* Refactor CI tests (#2504)
* Update APK versions (#2505)
* Export whisper distil-large-v3 and distil-large-v3.5 to sherpa-onnx (#2506)
* Support specifying pronunciations of phrases in Chinese TTS. (#2507)
* fix(flutter): fix unicode problem in windows path (#2508)
* feat: add punctuation C++ API (#2510)
* Fix ctrl+c may lead to coredump (#2511)
* Add kitten tts nano v0.2 (#2512)
* Scripts to generate tts samples (#2513)
* Add tdt duration to APIs (#2514)
* Support 16KB page size for Android (#2520)
* Split sherpa-onnx Python package (#2521)
* Fix kokoro tts for punctuations (#2522)

## 1.12.9

* Add more piper tts models (#2480)
* Fix ASR for UE (#2483)
* push to maven center (#2463)
* Specify ABIs when building APKs (#2488)
* Add more debug info for vits tts (#2491)
* Add Swift API for computing speaker embeddings (#2492)
* Alex/feat add python example (#2490)
* Support TDT transducer decoding (#2495)
* Fix java test (#2496)
* Refactor Swift API (#2493)
* add TtsReader app to README.md (#2498)
* Export https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3 to sherpa-onnx (#2500)
* Fix building apk (#2499)

## 1.12.8

* Expose JNI to compute probability of chunk in VAD (#2433)
* Add https://huggingface.co/Banafo/Kroko-ASR (#2453)
* Add APIs for Online NeMo CTC models (#2454)
* Export https://github.com/KittenML/KittenTTS to sherpa-onnx (#2456)
* Fix punctuations in kokoro tts. (#2458)
* Limit number of tokens in fire red asr decoding. (#2459)
* Add C++ runtime for kitten-tts (#2460)
* Add Kotlin and Java API for KittenTTS (#2461)
* Add Android TTS Engine APK for KittenTTS (#2465)
* Add Python API for KittenTTS. (#2466)
* Add C API for KittenTTS (#2467)
* Add CXX API for KittenTTS (#2469)
* Add JavaScript API (node-addon) for KittenTTS (#2470)
* Add JavaScript API (WebAssembly) for KittenTTS (#2471)
* Add Pascal API for KittenTTS (#2474)
* Add Dart API for KittenTTS (#2475)
* Add Swift API for KittenTTS (#2476)
* Add C# API for KittenTTS (#2477)
* Add Go API for KittenTTS (#2478)

## 1.12.7

* Support Portuguese and German ASR models from NeMo (#2394)
* Support returning the current speech segment for VAD. (#2397)
* Add more piper tts polish models (#2403)
* Support VAD+ASR for WearOS (#2404)
* Support test long audio with streaming-model & vad (#2405)
* Fix typo in sherpa-onnx-vad-with-online-asr.cc (#2407)
* Add tail padding for sherpa-onnx-vad-with-online-asr (#2408)
* Add more French TTS models (#2424)
* Add more piper tts models (#2425)
* Implement max_symbols_per_frame for GigaAM2 accurate decoding since model uses char tokens instead of BPE. (#2423)
* Fix GigaAM transducer encoder output length data type (#2426)
* Add friendly log messages for Android and HarmonyOs TTS users. (#2427)
* Fix setGraph in OnlineCtcFstDecoderConfig Java API (#2411)


## 1.12.6

* Support silero-vad v4 exported by k2-fsa (#2372)
* Add C++ and Python support for ten-vad (#2377)
* Fix compile errors for Linux (#2378)
* Add C API for ten-vad (#2379)
* Add CXX API examples for ten-vad. (#2380)
* Add JavaScript (WebAssembly) API for ten-vad (#2382)
* Add JavaScript (node-addon) API for ten-vad (#2383)
* Add Go API for ten-vad (#2384)
* Add C# API for ten-vad (#2385)
* Add Dart API for ten-vad (#2386)
* Add Swift API for ten-vad (#2387)
* Add Pascal API for ten-vad (#2388)
* Add Java/Kotlin API and Android support for ten-vad (#2389)

## 1.12.5

* Fix typo CMAKE_EXECUTBLE_LINKER_FLAGS -> CMAKE_EXECUTABLE_LINKER_FLAGS (#2344)
* Fix testing dart packages (#2345)
* fix(canary): use dynamo export, single input_ids and avoid 0/1 specialization (#2348)
* Fix TTS for Unreal Engine (#2349)
* Update readme to include https://github.com/mawwalker/stt-server (#2350)
* Add meta data to NeMo canary ONNX models (#2351)
* Update README to include https://github.com/bbeyondllove/asr_server (#2353)
* Add C++ runtime and Python API for NeMo Canary models (#2352)
* Add C/CXX/JavaScript API for NeMo Canary models (#2357)
* Add Java and Kotlin API for NeMo Canary models (#2359)
* Upload fp16 onnx model files for FireRedASR (#2360)
* Fix nemo feature normalization in test code (#2361)
* Refactor exporting NeMo models (#2362)
* Add LODR support to online and offline recognizers (#2026)
* Add CXX examples for NeMo TDT ASR. (#2363)
* Add Pascal/Go/C#/Dart API for NeMo Canary ASR models (#2367)

## 1.12.4

* Refactor release scripts. (#2323)
* Add TTS engine APKs for more models (#2327)
* Fix static link without tts (#2328)
* Fix VAD+ASR C++ example. (#2335)
* Add sherpa-onnx-streaming-zipformer-zh-int8-2025-06-30 to android ASR apk (#2336)
* Support non-streaming zipformer CTC ASR models (#2340)
* Support linux aarch64 for Dart and Flutter (#2342)

## 1.12.3

* Show CMake debug information. (#2316)
* Remove portaudio-go in Go API examples. (#2317)
* Support Xipformer CTC ASR with whisper features. (#2319)
* Support Zipformer transducer ASR with whisper features. (#2321)

## 1.12.2

* Fix CI for windows (#2279)
* Add jar for Java 24. (#2280)
* Add Python API for source separation (#2283)
* Add link to huggingface space for source separation. (#2284)
* Fix isspace on windows in debug build (#2042)
* Update wasm/vad-asr/assets/README.md for more clear (#2297)
* Update TTS Engine APK to support multi-lang (#2294)
* Add scripts for exporting Piper TTS models to sherpa-onnx (#2299)
* Update sherpa-onnx-shared.pc.in (#2300)
* Fixes #2172 (#2301)
* Refactor kokoro export (#2302)
* Fix building for Pascal (#2305)
* Support extra languages in multi-lang kokoro tts (#2303)
* Update readme to include BreezeApp from MediaTek Research. (#2313)
* Add API to get version information (#2309)


## 1.12.1

* Use jlong explicitly in jni. (#2229)
* Fix building RKNN wheels (#2233)
* Fix publishing binaries for RKNN (#2234)
* Export spleeter model to onnx for source separation (#2237)
* Add C++ runtime for spleeter about source separation (#2242)
* Add include headers for __ANDROID_API__,__OHOS__ (#2251)
* JAVA-API: Manual Library Loading Support for Restricted Environments (#2253)
* Build APK with replace.fst (#2254)
* repair rknn wheels (#2257)
* Update kaldi-native-fbank. (#2259)
* Fix building sherpa-onnx (#2262)
* Fix building MFC examples (#2263)
* Add UVR models for source separation. (#2266)
* move portaudio common record code to microphone (#2264)
* fixed mfc build error (#2267)
* Add C++ support for UVR models (#2269)
* Export nvidia/canary-180m-flash to sherpa-onnx (#2272)
* Update utils.dart (#2275)
* Fix rknn for multi-threads (#2274)
* Fix 32-bit arm CI (#2276)

## 1.12.0

* Fix building wheels for macOS (#2192)
* Show verbose logs in homophone replacer (#2194)
* Fix displaying streaming speech recognition results for Python. (#2196)
* Add real-time speech recognition example for SenseVoice. (#2197)
* docs: add Open-XiaoAI KWS project (#2198)
* Add C++ example for streaming ASR with SenseVoice. (#2199)
* Add C++ example for real-time ASR with nvidia/parakeet-tdt-0.6b-v2. (#2201)
* Add a link to YouTube video including sherpa-onnx. (#2202)
* Support sending is_eof for online websocket server. (#2204)
* Add alsa-based streaming ASR example for sense voice. (#2207)
* Support homophone replacer in Android asr demo. (#2210)
* Add Go implementation of the TTS generation callback (#2213)
* Add Android demo for real-time ASR with non-streaming ASR models. (#2214)
* Expose dither for JNI (#2215)
* Add nodejs example for parakeet-tdt-0.6b-v2. (#2219)
* Add script to build APK for simulated-streaming-asr. (#2220)


## 1.11.5

* export parakeet-tdt-0.6b-v2 to sherpa-onnx (#2180)
* Add C++ runtime for parakeet-tdt-0.6b-v2. (#2181)
* Avoid NaN in feature normalization. (#2186)

## 1.11.4

* Disable strict hotword matching mode for offline transducer (#1837)
* Comment refinement: Add note about vocoder file for matcha TTS config (#2106)
* Fix a typo in the JNI for Android. (#2108)
* Generate subtitles with FireRedAsr models (#2112)
* Use manylinux_2_28_x86_64 to build linux gpu for sherpa-onnx (#2123)
* Support running sherpa-onnx with RK NPU on Android (#2124)
* Fix building for HarmonyOS (#2125)
* cmake build, configurable from env (#2115)
* Expose dither in python API (#2127)
* Add support for GigaAM-CTC-v2 (#2135)
* Support Giga AM transducer V2 (#2136)
* Export kokoro 1.0 int8 models (#2137)
* Upload more onnx ASR models (#2141)
* Fix building for open harmonyOS (#2142)
* online-transducer: reset the encoder toghter with 2 previous output symbols (non-blank) (#2129)
* Fix punctuations for kokoro tts 1.1-zh. (#2146)
* Fix setting OnlineModelConfig in Java API (#2147)
* Support decoding multiple streams in Java API. (#2149)
* Support replacing homonphonic phrases (#2153)
* Add C and CXX API for homophone replacer (#2156)
* Add JavaScript API (WASM) for homophone replacer (#2157)
* Add JavaScript API (node-addon) for homophone replacer (#2158)
* Fix building without TTS (#2159)
* Add homophone replacer example for Python API. (#2161)
* More fix for building without tts (#2162)
* Add Swift API for homophone replacer. (#2164)
* Add C# API for homophone replacer (#2165)
* Add Kotlin and Java API for homophone replacer (#2166)
* Add Dart API for homophone replacer (#2167)
* Add Go API for homophone replacer (#2168)

## 1.11.3

* fix vits dict dir config (#2036)
* fix case (#2037)
* Fix building wheels for RKNN (#2041)
* Change scale factor to 32767 (#2056)
* Fix length scale for kokoro tts (#2060)
* Allow building repository as CMake subdirectory (#2059)
* Export silero_vad v4 to RKNN (#2067)
* fix dml with preinstall ort (#2066)
* Fix building aar to include speech denoiser (#2069)
* Add CXX API for VAD (#2077)
* Add C++ runtime for silero_vad with RKNN (#2078)
* Refactor rknn code (#2079)
* Fix building for android (#2081)
* Add C++ and Python API for Dolphin CTC models (#2085)
* Add Kotlin and Java API for Dolphin CTC models (#2086)
* Add C and CXX API for Dolphin CTC models (#2088)
* Preserve more context after endpointing in transducer (#2061)
* Add C# API for Dolphin CTC models (#2089)
* Add Go API for Dolphin CTC models (#2090)
* Add Swift API for Dolphin CTC models (#2091)
* Add Javascript (WebAssembly) API for Dolphin CTC models (#2093)
* Add Javascript (node-addon) API for Dolphin CTC models (#2094)
* Add Dart API for Dolphin CTC models (#2095)
* Add Pascal API for Dolphin CTC models (#2096)

## 1.11.2

* Fix CI (#2016)
* Publish jar for more java versions (#2017)
* add alsa example for vad+offline asr (#2020)
* Support cuda12 and cudnn8 for Linux aarch64. (#2021)
* Update README to include more projects using sherpa-onnx (#2022)
* Fix a bug in vad.reset() (#2023)
* Fix Matcha + vocos for Android (#2024)
* Fix crash in Android tts engine demo. (#2029)
* Fix build script: add 'cd build' after 'mkdir build' to ensure the correct working directory for CMake (#2033)
* fix static linking (#2032)

## 1.11.1

* Export vocos to sherpa-onnx (#2012)
* Add C++ runtime for vocos (#2014)

## 1.11.0

* Fix building wheels for Python 3.7 (#1933)
* Add Kotlin and Java API for online punctuation models (#1936)
* Add Kokoro v1.1-zh (#1942)
* Support RKNN for Zipformer CTC models. (#1948)
* Add transducer modified_beam_search for RKNN. (#1949)
* Update README to include projects that is using sherpa-onnx (#1956)
* Limit number of tokens per second for whisper. (#1958)
* Ebranchformer (#1951)
* Test using sherpa-onnx as a cmake subproject (#1961)
* Add C++ demo for VAD+non-streaming ASR (#1964)
* Export gtcrn models to sherpa-onnx (#1975)
* c-api add wave write to buffer. (#1962)
* add SherpaOnnxOfflineRecognizerSetConfig binding for go, and optimize the new/free for C.struct_SherpaOnnxOfflineRecognizerConfig ptr (#1976)
* Add C++ runtime for speech enhancement GTCRN models (#1977)
* Add Python API for speech enhancement GTCRN models (#1978)
* Add C API for speech enhancement GTCRN models (#1984)
* Add CXX API for speech enhancement GTCRN models (#1986)
* Add Swift API for speech enhancement GTCRN models (#1989)
* Add C# API for speech enhancement GTCRN models (#1990)
* Add Go API for speech enhancement GTCRN models (#1991)
* Add Pascal API for speech enhancement GTCRN models (#1992)
* Add Dart API for speech enhancement GTCRN models (#1993)
* Add JavaScript (node-addon) API for speech enhancement GTCRN models (#1996)
* Add WebAssembly (WASM) for speech enhancement GTCRN models (#2002)
* Add JavaScript API (wasm) for speech enhancement GTCRN models (#2007)
* Add Kotlin API for speech enhancement GTCRN models (#2008)
* Add Java API for speech enhancement GTCRN models (#2009)





## 1.10.46

* Fix kokoro lexicon. (#1886)
* speaker-identification-with-vad-non-streaming-asr.py Lack of support for sense_voice. (#1884)
* Fix generating Chinese lexicon for Kokoro TTS 1.0 (#1888)
* Reduce vad-whisper-c-api example code. (#1891)
* JNI Exception Handling (#1452)
* Fix #1901: UnicodeEncodeError running export_bpe_vocab.py (#1902)
* Fix publishing pre-built windows libraries (#1905)
* Fixing Whisper Model Token Normalization (#1904)
* feat: add mic example for better compatibility (#1909)
* Add onnxruntime 1.18.1 for Linux aarch64 GPU (#1914)
* Add C++ API for streaming zipformer ASR on RK NPU (#1908)
* change [1<<28] to [1<<10], to fix build issues on GOARCH=386 that [1<<28] too large (#1916)
* Flutter Config toJson/fromJson (#1893)
* Fix publishing linux pre-built artifacts (#1919)
* go.mod set to use go 1.17, and use unsafe.Slice to optimize the code (#1920)
* fix: AddPunct panic for Go(#1921)
* Fix publishing macos pre-built artifacts (#1922)
* Minor fixes for rknn (#1925)
* Build wheels for rknn linux aarch64 (#1928)

## 1.10.45

* [update] fixed bug: create golang instance succeed while the c struct create failed (#1860)
* fixed typo in RTF calculations (#1861)
* Export FireRedASR to sherpa-onnx. (#1865)
* Add C++ and Python API for FireRedASR AED models (#1867)
* Add Kotlin and Java API for FireRedAsr AED model (#1870)
* Add C API for FireRedAsr AED model. (#1871)
* Add CXX API for FireRedAsr (#1872)
* Add JavaScript API (node-addon) for FireRedAsr (#1873)
* Add JavaScript API (WebAssembly) for FireRedAsr model. (#1874)
* Add C# API for FireRedAsr Model (#1875)
* Add C# API for FireRedAsr Model (#1875)
* Add Swift API for FireRedAsr AED Model (#1876)
* Add Dart API for FireRedAsr AED Model (#1877)
* Add Go API for FireRedAsr AED Model (#1879)
* Add Pascal API for FireRedAsr AED Model (#1880)

## 1.10.44

* Export MatchaTTS fa-en model to sherpa-onnx (#1832)
* Add C++ support for MatchaTTS models not from icefall. (#1834)
* OfflineRecognizer supports create stream with hotwords (#1833)
* Add PengChengStarling models to sherpa-onnx (#1835)
* Support specifying voice in espeak-ng for kokoro tts models. (#1836)
* Fix: made print sherpa_onnx_loge when it is in debug mode (#1838)
* Add Go API for audio tagging (#1840)
* Fix CI (#1841)
* Update readme to contain links for pre-built Apps (#1853)
* Modify the model used (#1855)
* Flutter OnlinePunctuation (#1854)
* Fix spliting text by languages for kokoro tts. (#1849)

## 1.10.43

* Add MFC example for Kokoro TTS 1.0 (#1815)
* Update sherpa-onnx-tts.js VitsModelConfig.model can be none (#1817)
* Fix passing gb2312 encoded strings to tts on Windows (#1819)
* Support scaling the duration of a pause in TTS. (#1820)
* Fix building wheels for linux aarch64. (#1821)
* Fix CI for Linux aarch64. (#1822)

## 1.10.42

* Fix publishing wheels (#1746)
* Update README to include https://github.com/xinhecuican/QSmartAssistant (#1755)
* Add Kokoro TTS to MFC examples (#1760)
* Refactor node-addon C++ code. (#1768)
* Add keyword spotter C API for HarmonyOS (#1769)
* Add ArkTS API for Keyword spotting. (#1775)
* Add Flutter example for Kokoro TTS (#1776)
* Initialize the audio session for iOS ASR example (#1786)
* Fix: Prepend 0 to tokenization to prevent word skipping for Kokoro. (#1787)
* Export Kokoro 1.0 to sherpa-onnx (#1788)
* Add C++ and Python API for Kokoro 1.0 multilingual TTS model (#1795)
* Add Java and Koltin API for Kokoro TTS 1.0 (#1798)
* Add Android demo for Kokoro TTS 1.0 (#1799)
* Add C API for Kokoro TTS 1.0 (#1801)
* Add CXX API for Kokoro TTS 1.0 (#1802)
* Add Swift API for Kokoro TTS 1.0 (#1803)
* Add Go API for Kokoro TTS 1.0 (#1804)
* Add C# API for Kokoro TTS 1.0 (#1805)
* Add Dart API for Kokoro TTS 1.0 (#1806)
* Add Pascal API for Kokoro TTS 1.0 (#1807)
* Add JavaScript API (node-addon) for Kokoro TTS 1.0 (#1808)
* Add JavaScript API (WebAssembly) for Kokoro TTS 1.0 (#1809)
* Add Flutter example for Kokoro TTS 1.0 (#1810)
* Add iOS demo for Kokoro TTS 1.0 (#1812)
* Add HarmonyOS demo for Kokoro TTS 1.0 (#1813)

## 1.10.41

* Fix UI for Android TTS Engine. (#1735)
* Add iOS TTS example for MatchaTTS (#1736)
* Add iOS example for Kokoro TTS (#1737)
* Fix dither binding in Pybind11 to ensure independence from high_freq in FeatureExtractorConfig (#1739)
* Fix keyword spotting. (#1689)
* Update readme to include https://github.com/hfyydd/sherpa-onnx-server (#1741)
* Reduce vad-moonshine-c-api example code. (#1742)
* Support Kokoro TTS for HarmonyOS. (#1743)

## 1.10.40

* Fix building wheels (#1703)
* Export kokoro to sherpa-onnx (#1713)
* Add C++ and Python API for Kokoro TTS models. (#1715)
* Add C API for Kokoro TTS models (#1717)
* Fix style issues (#1718)
* Add C# API for Kokoro TTS models (#1720)
* Add Swift API for Kokoro TTS models (#1721)
* Add Go API for Kokoro TTS models (#1722)
* Add Dart API for Kokoro TTS models (#1723)
* Add Pascal API for Kokoro TTS models (#1724)
* Add JavaScript API (node-addon) for Kokoro TTS models (#1725)
* Add JavaScript (WebAssembly) API for Kokoro TTS models. (#1726)
* Add Koltin and Java API for Kokoro TTS models (#1728)
* Update README.md for KWS to not use git lfs. (#1729)




## 1.10.39

* Fix building without TTS (#1691)
* Add README for android libs. (#1693)
* Fix: export-onnx.py(expected all tensors to be on the same device) (#1699)
* Fix passing strings from C# to C. (#1701)

## 1.10.38

* Fix initializing TTS in Python. (#1664)
* Remove spaces after punctuations for TTS (#1666)
* Add constructor fromPtr() for all flutter class with factory ctor. (#1667)
* Add Kotlin API for Matcha-TTS models. (#1668)
* Support Matcha-TTS models using espeak-ng (#1672)
* Add Java API for Matcha-TTS models. (#1673)
* Avoid adding tail padding for VAD in generate-subtitles.py (#1674)
* Add C API for MatchaTTS models (#1675)
* Add CXX API for MatchaTTS models (#1676)
* Add JavaScript API (node-addon-api) for MatchaTTS models. (#1677)
* Add HarmonyOS examples for MatchaTTS. (#1678)
* Upgraded to .NET 8 and made code style a little more internally consistent. (#1680)
* Update workflows to use .NET 8.0 also. (#1681)
* Add C# and JavaScript (wasm) API for MatchaTTS models (#1682)
* Add Android demo for MatchaTTS models. (#1683)
* Add Swift API for MatchaTTS models. (#1684)
* Add Go API for MatchaTTS models (#1685)
* Add Pascal API for MatchaTTS models. (#1686)
* Add Dart API for MatchaTTS models (#1687)

## 1.10.37

* Add new tts models for Latvia and Persian+English (#1644)
* Add a byte-level BPE Chinese+English non-streaming zipformer model (#1645)
* Support removing invalid utf-8 sequences. (#1648)
* Add TeleSpeech CTC to non_streaming_server.py (#1649)
* Fix building macOS libs (#1656)
* Add Go API for Keyword spotting (#1662)
* Add Swift online punctuation (#1661)
* Add C++ runtime for Matcha-TTS (#1627)

## 1.10.36

* Update AAR version in Android Java demo (#1618)
* Support linking onnxruntime statically for Android (#1619)
* Update readme to include Open-LLM-VTuber (#1622)
* Rename maxNumStences to maxNumSentences (#1625)
* Support using onnxruntime 1.16.0 with CUDA 11.4 on Jetson Orin NX (Linux arm64 GPU). (#1630)
* Update readme to include jetson orin nx and nano b01 (#1631)
* feat: add checksum action (#1632)
* Support decoding with byte-level BPE (bbpe) models. (#1633)
* feat: enable c api for android ci (#1635)
* Update README.md (#1640)
* SherpaOnnxVadAsr: Offload runSecondPass to background thread for improved real-time audio processing (#1638)
* Fix GitHub actions. (#1642)


## 1.10.35

* Add missing changes about speaker identfication demo for HarmonyOS (#1612)
* Provide sherpa-onnx.aar for Android (#1615)
* Use aar in Android Java demo. (#1616)

## 1.10.34

* Fix building node-addon package (#1598)
* Update doc links for HarmonyOS (#1601)
* Add on-device real-time ASR demo for HarmonyOS (#1606)
* Add speaker identification APIs for HarmonyOS (#1607)
* Add speaker identification demo for HarmonyOS (#1608)
* Add speaker diarization API for HarmonyOS. (#1609)
* Add speaker diarization demo for HarmonyOS (#1610)

## 1.10.33

* Add non-streaming ASR support for HarmonyOS. (#1564)
* Add streaming ASR support for HarmonyOS. (#1565)
* Fix building for Android (#1568)
* Publish `sherpa_onnx.har` for HarmonyOS (#1572)
* Add VAD+ASR demo for HarmonyOS (#1573)
* Fix publishing har packages for HarmonyOS (#1576)
* Add CI to build HAPs for HarmonyOS (#1578)
* Add microphone demo about VAD+ASR for HarmonyOS (#1581)
* Fix getting microphone permission for HarmonyOS VAD+ASR example (#1582)
* Add HarmonyOS support for text-to-speech. (#1584)
* Fix: support both old and new websockets request headers format (#1588)
* Add on-device tex-to-speech (TTS) demo for HarmonyOS (#1590)

## 1.10.32

* Support cross-compiling for HarmonyOS (#1553)
* HarmonyOS support for VAD. (#1561)
* Fix publishing flutter iOS app to appstore (#1563).

## 1.10.31

* Publish pre-built wheels for Python 3.13 (#1485)
* Publish pre-built macos xcframework (#1490)
* Fix reading tokens.txt on Windows. (#1497)
* Add two-pass ASR Android APKs for Moonshine models. (#1499)
* Support building GPU-capable sherpa-onnx on Linux aarch64. (#1500)
* Publish pre-built wheels with CUDA support for Linux aarch64. (#1507)
* Export the English TTS model from MeloTTS (#1509)
* Add Lazarus example for Moonshine models. (#1532)
* Add isolate_tts demo (#1529)
* Add WebAssembly example for VAD + Moonshine models. (#1535)
* Add Android APK for streaming Paraformer ASR (#1538)
* Support static build for windows arm64. (#1539)
* Use xcframework for Flutter iOS plugin to support iOS simulators.

## 1.10.30

* Fix building node-addon for Windows x86. (#1469)
* Begin to support https://github.com/usefulsensors/moonshine (#1470)
* Publish pre-built JNI libs for Linux aarch64 (#1472)
* Add C++ runtime and Python APIs for Moonshine models (#1473)
* Add Kotlin and Java API for Moonshine models (#1474)
* Add C and C++ API for Moonshine models (#1476)
* Add Swift API for Moonshine models. (#1477)
* Add Go API examples for adding punctuations to text. (#1478)
* Add Go API for Moonshine models (#1479)
* Add JavaScript API for Moonshine models (#1480)
* Add Dart API for Moonshine models. (#1481)
* Add Pascal API for Moonshine models (#1482)
* Add C# API for Moonshine models. (#1483)

## 1.10.29

* Add Go API for offline punctuation models (#1434)
* Support https://huggingface.co/Revai/reverb-diarization-v1 (#1437)
* Add more models for speaker diarization (#1440)
* Add Java API example for hotwords. (#1442)
* Add java android demo (#1454)
* Add C++ API for streaming ASR. (#1455)
* Add C++ API for non-streaming ASR (#1456)
* Handle NaN embeddings in speaker diarization. (#1461)
* Add speaker identification with VAD and non-streaming ASR using ALSA (#1463)
* Support GigaAM CTC models for Russian ASR (#1464)
* Add GigaAM NeMo transducer model for Russian ASR (#1467)

## 1.10.28

* Fix swift example for generating subtitles. (#1362)
* Allow more online models to load tokens file from the memory (#1352)
* Fix CI errors introduced by supporting loading keywords from buffers (#1366)
* Fix running MeloTTS models on GPU. (#1379)
* Support Parakeet models from NeMo (#1381)
* Export Pyannote speaker segmentation models to onnx (#1382)
* Support Agglomerative clustering. (#1384)
* Add Python API for clustering (#1385)
* support whisper turbo (#1390)
* context_state is not set correctly when previous context is passed after reset (#1393)
* Speaker diarization example with onnxruntime Python API (#1395)
* C++ API for speaker diarization (#1396)
* Python API for speaker diarization. (#1400)
* C API for speaker diarization (#1402)
* docs(nodejs-addon-examples): add guide for pnpm user (#1401)
* Go API for speaker diarization (#1403)
* Swift API for speaker diarization (#1404)
* Update readme to include more external projects using sherpa-onnx (#1405)
* C# API for speaker diarization (#1407)
* JavaScript API (node-addon) for speaker diarization (#1408)
* WebAssembly exmaple for speaker diarization (#1411)
* Handle audio files less than 10s long for speaker diarization. (#1412)
* JavaScript API with WebAssembly for speaker diarization (#1414)
* Kotlin API for speaker diarization (#1415)
* Java API for speaker diarization (#1416)
* Dart API for speaker diarization (#1418)
* Pascal API for speaker diarization (#1420)
* Android JNI support for speaker diarization (#1421)
* Android demo for speaker diarization (#1423)

## 1.10.27

* Add non-streaming ONNX models for Russian ASR (#1358)
* Fix building Flutter TTS examples for Linux (#1356)
* Support passing utf-8 strings from JavaScript to C++. (#1355)
* Fix sherpa_onnx.go to support returning empty recognition results (#1353)

## 1.10.26

* Add links to projects using sherpa-onnx. (#1345)
* Support lang/emotion/event results from SenseVoice in Swift API. (#1346)
* Support specifying max speech duration for VAD. (#1348)
* Add APIs about max speech duration in VAD for various programming languages (#1349)

## 1.10.25

* Allow tokens and hotwords to be loaded from buffered string driectly (#1339)
* Fix computing features for CED audio tagging models. (#1341)
* Preserve previous result as context for next segment (#1335)
* Add Python binding for online punctuation models (#1312)
* Fix vad.Flush(). (#1329)
* Fix wasm app for streaming paraformer (#1328)
* Build websocket related binaries for embedded systems. (#1327)
* Fixed the C api calls and created the TTS project file (#1324)
* Re-implement LM rescore for online transducer (#1231)

## 1.10.24

* Add VAD and keyword spotting for the Node package with WebAssembly (#1286)
* Fix releasing npm package and fix building Android VAD+ASR example (#1288)
* add Tokens []string, Timestamps []float32, Lang string, Emotion string, Event string (#1277)
* add vad+sense voice example for C API (#1291)
* ADD VAD+ASR example for dart with CircularBuffer. (#1293)
* Fix VAD+ASR example for Dart API. (#1294)
* Avoid SherpaOnnxSpeakerEmbeddingManagerFreeBestMatches freeing null. (#1296)
* Fix releasing wasm app for vad+asr (#1300)
* remove extra files from linux/macos/windows jni libs (#1301)
* two-pass Android APK for SenseVoice (#1302)
* Downgrade flutter sdk versions. (#1305)
* Reduce onnxruntime log output. (#1306)
* Provide prebuilt .jar files for different java versions. (#1307)


## 1.10.23

* flutter: add lang, emotion, event to OfflineRecognizerResult (#1268)
* Use a separate thread to initialize models for lazarus examples. (#1270)
* Object pascal examples for recording and playing audio with portaudio. (#1271)
* Text to speech API for Object Pascal. (#1273)
* update kotlin api for better release native object and add user-friendly apis. (#1275)
* Update wave-reader.cc to support 8/16/32-bit waves (#1278)
* Add WebAssembly for VAD (#1281)
* WebAssembly example for VAD + Non-streaming ASR (#1284)

## 1.10.22

* Add Pascal API for reading wave files (#1243)
* Pascal API for streaming ASR (#1246)
* Pascal API for non-streaming ASR (#1247)
* Pascal API for VAD (#1249)
* Add more C API examples (#1255)
* Add emotion, event of SenseVoice. (#1257)
* Support reading multi-channel wave files with 8/16/32-bit encoded samples (#1258)
* Enable IPO only for Release build. (#1261)
* Add Lazarus example for generating subtitles using Silero VAD with non-streaming ASR (#1251)
* Fix looking up OOVs in lexicon.txt for MeloTTS models. (#1266)


## 1.10.21

* Fix ffmpeg c api example (#1185)
* Fix splitting sentences for MeloTTS (#1186)
* Non-streaming WebSocket client for Java. (#1190)
* Fix copying asset files for flutter examples. (#1191)
* Add Chinese+English tts example for flutter (#1192)
* Add speaker identification and verification exmaple for Dart API (#1194)
* Fix reading non-standard wav files. (#1199)
* Add ReazonSpeech Japanese pre-trained model (#1203)
* Describe how to add new words for MeloTTS models (#1209)
* Remove libonnxruntime_providers_cuda.so as a dependency. (#1210)
* Fix setting SenseVoice language. (#1214)
* Support passing TTS callback in Swift API (#1218)
* Add MeloTTS example for ios (#1223)
* Add online punctuation and casing prediction model for English language (#1224)
* Fix python two pass ASR examples (#1230)
* Add blank penalty for various language bindings

## 1.10.20

* Add Dart API for audio tagging
* Add Dart API for adding punctuations to text

## 1.10.19

* Prefix all C API functions with SherpaOnnx

## 1.10.18

* Fix the case when recognition results contain the symbol `"`. It caused
  issues when converting results to a json string.

## 1.10.17

* Support SenseVoice CTC models.
* Add Dart API for keyword spotter.

## 1.10.16

* Support zh-en TTS model from MeloTTS.

## 1.10.15

* Downgrade onnxruntime from v1.18.1 to v1.17.1

## 1.10.14

* Support whisper large v3
* Update onnxruntime from v1.18.0 to v1.18.1
* Fix invalid utf8 sequence from Whisper for Dart API.

## 1.10.13

* Update onnxruntime from 1.17.1 to 1.18.0
* Add C# API for Keyword spotting

## 1.10.12

* Add Flush to VAD so that the last speech segment can be detected. See also
  https://github.com/k2-fsa/sherpa-onnx/discussions/1077#discussioncomment-9979740

## 1.10.11

* Support the iOS platform for Flutter.

## 1.10.10

* Build sherpa-onnx into a single shared library.

## 1.10.9

* Fix released packages. piper-phonemize was not included in v1.10.8.

## 1.10.8

* Fix released packages. There should be a lib directory.

## 1.10.7

* Support Android for Flutter.

## 1.10.2

* Fix passing C# string to C++

## 1.10.1

* Enable to stop TTS generation

## 1.10.0

* Add inverse text normalization

## 1.9.30

* Add TTS

## 1.9.29

* Publish with CI

## 0.0.3

* Fix path separator on Windows.

## 0.0.2

* Support specifying lib path.

## 0.0.1

* Initial release.
