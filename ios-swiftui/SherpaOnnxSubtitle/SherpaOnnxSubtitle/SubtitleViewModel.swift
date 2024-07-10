//
//  SubtitleViewModel.swift
//  SherpaOnnxSubtitle
//
//  Created by knight on 2023/9/23.
//

import AVFoundation
import PhotosUI
import SwiftUI

enum LoadState {
    case initial
    case loading
    case loaded(Audio)
    case done
    case failed
}

class SubtitleViewModel: ObservableObject {
    var modelType = "whisper"
    let sampleRate = 16000

    var modelConfig: SherpaOnnxOfflineModelConfig?
    // modelType = "paraformer"

    var recognizer: SherpaOnnxOfflineRecognizer?

    var vadModelConfig: SherpaOnnxVadModelConfig?
    var vad: SherpaOnnxVoiceActivityDetectorWrapper?

    @Published var loadState: LoadState = .initial

    @Published var selectedItem: PhotosPickerItem? = nil

    @Published var importNow: Bool = false {
        didSet {
            loadState = .loading
        }
    }

    @Published var exportNow: Bool = false

    var srtName: String = "unknown.srt"
    var content: String = ""

    var srtDocument: Document {
        let content = content.data(using: .utf8)
        return Document(data: content)
    }

    var hasAudio: Bool {
        return selectedItem != nil
    }

    init() {
        if modelType == "whisper" {
            // for English
            self.modelConfig = getNonStreamingWhisperTinyEn()
        } else if modelType == "paraformer" {
            // for Chinese
            self.modelConfig = getNonStreamingZhParaformer20230914()
        } else {
            print("Please specify a supported modelType \(modelType)")
            return
        }

        let featConfig = sherpaOnnxFeatureConfig(
            sampleRate: sampleRate,
            featureDim: 80
        )

        guard let modelConfig else {
            return
        }

        var config = sherpaOnnxOfflineRecognizerConfig(
            featConfig: featConfig,
            modelConfig: modelConfig
        )

        recognizer = SherpaOnnxOfflineRecognizer(config: &config)

        let sileroVadConfig = sherpaOnnxSileroVadModelConfig(
            model: getResource("silero_vad", "onnx")
        )

        self.vadModelConfig = sherpaOnnxVadModelConfig(sileroVad: sileroVadConfig)
        guard var vadModelConfig else {
            return
        }
        vad = SherpaOnnxVoiceActivityDetectorWrapper(
            config: &vadModelConfig, buffer_size_in_seconds: 120
        )
    }

    func restoreState() {
        loadState = .initial
    }

    func generateSRT(from file: URL) {
        print("gen srt from: \(file)")
        content = ""

        // restore state
        defer {
            loadState = .done
        }
        guard let recognizer else {
            return
        }
        guard let vadModelConfig else {
            return
        }

        guard let vad else {
            return
        }

        do {
            let audioFile = try AVAudioFile(forReading: file)
            let audioFormat = audioFile.processingFormat
            assert(audioFormat.sampleRate == Double(sampleRate))
            assert(audioFormat.channelCount == 1)
            assert(audioFormat.commonFormat == AVAudioCommonFormat.pcmFormatFloat32)

            let audioFrameCount = UInt32(audioFile.length)
            let audioFileBuffer = AVAudioPCMBuffer(pcmFormat: audioFormat, frameCapacity: audioFrameCount)

            try audioFile.read(into: audioFileBuffer!)
            var array: [Float]! = audioFileBuffer?.array()

            let windowSize = Int(vadModelConfig.silero_vad.window_size)

            var segments: [SpeechSegment] = []

            while array.count > windowSize {
                // todo(fangjun): avoid extra copies here
                vad.acceptWaveform(samples: [Float](array[0 ..< windowSize]))
                array = [Float](array[windowSize ..< array.count])

                while !vad.isEmpty() {
                    let s = vad.front()
                    vad.pop()
                    let result = recognizer.decode(samples: s.samples)

                    segments.append(
                        SpeechSegment(
                            start: Float(s.start) / Float(sampleRate),
                            duration: Float(s.samples.count) / Float(sampleRate),
                            text: result.text
                        ))

                    print(segments.last!)
                }
            }
            content = zip(segments.indices, segments).map { index, element in
                "\(index + 1)\n\(element)"
            }.joined(separator: "\n\n")
        } catch {
            print("error: \(error.localizedDescription)")
        }
        exportNow = true

        let last = file.lastPathComponent
        srtName = "\(last).srt"
    }
}
