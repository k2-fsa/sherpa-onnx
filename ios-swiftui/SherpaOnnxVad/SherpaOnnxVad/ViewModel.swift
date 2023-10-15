//
//  ViewModel.swift
//  SherpaOnnxVad
//
//  Created by knight on 2023/10/11.
//

import AVFoundation
import Foundation
import SwiftUI

enum ListenStatus {
    case stopped
    case listening
    
    // instruct what to do next
    var instruction: String {
        switch self {
        case .stopped: return "START"
        case .listening: return "STOP"
        }
    }
}

class ViewModel: ObservableObject {
    var audioEngine: AVAudioEngine = .init()
    private let inputBus: AVAudioNodeBus = 0
    private let outputBus: AVAudioNodeBus = 0
    private var sampleRate: Double = 16000
    private let bufferSize: AVAudioFrameCount = 1024
    private var inputFormat: AVAudioFormat!
    
    private var streamingData = false
    
    // VAD
    var vadModelConfig: SherpaOnnxVadModelConfig?
    var vad: SherpaOnnxVoiceActivityDetectorWrapper?


    @Published var listenStatus: ListenStatus = .stopped
    
    @Published var isSpeaking: Bool = false
    
    var indicatorColor: Color {
        isSpeaking ? .red : .black
    }
    
    var buttonInstruction: String {
        return listenStatus.instruction
    }
    
    init() {
        initVadModel()
    }
    
    private func initVadModel() {
        // https://github.com/snakers4/silero-vad/blob/563106ef8cfac329c8be5f9c5051cd365195aff9/utils_vad.py#L170
        let sileroVadConfig = sherpaOnnxSileroVadModelConfig(
            model: getResource("silero_vad", "onnx"),
            threshold: 0.5,
            minSilenceDuration: 0.1,
            minSpeechDuration: 0.25,
            windowSize: 512
        )

        self.vadModelConfig = sherpaOnnxVadModelConfig(sileroVad: sileroVadConfig)
        guard var vadModelConfig else {
            return
        }
        vad = SherpaOnnxVoiceActivityDetectorWrapper(
            config: &vadModelConfig, buffer_size_in_seconds: 120
        )
    }
    
    func toggle() {
        if listenStatus == .stopped {
            start()
            listenStatus = .listening
            return
        }
        
        if listenStatus == .listening {
            stop()
            listenStatus = .stopped
            return
        }
    }
    
    func start() {
        
        setupEngine()

        guard audioEngine.inputNode.inputFormat(forBus: inputBus).channelCount > 0 else {
            print("[AudioEngine]: No input is available.")
            streamingData = false
            return
        }
        
        do {
            try audioEngine.start()
            print("audio engine started")
        } catch {
            print("error: \(error.localizedDescription)")
        }
    }
    
    func stop() {
        audioEngine.stop()
        audioEngine.reset()
        audioEngine.inputNode.removeTap(onBus: inputBus)
        print("[AudioEngine] stopped now.")
    }

    private func setupEngine() {
        /// I don't know what the heck is happening under the hood, but if you don't call these next few lines in one closure your code will crash.
        /// Maybe it's threading issue?
        audioEngine.reset()
        let inputNode = audioEngine.inputNode
        inputFormat = inputNode.outputFormat(forBus: outputBus)
        inputNode.installTap(onBus: inputBus, bufferSize: bufferSize, format: inputFormat, block: { [weak self] buffer, time in
            self?.convert(buffer: buffer, time: time.audioTimeStamp.mSampleTime)
        })
        audioEngine.prepare()
    }
    
    private func handle(buffer: AVAudioPCMBuffer) {
        let voices = buffer.array()
        if voices.isEmpty {
            print("voice is empty.")
            return
        }
        
        guard let vadModelConfig else {
            return
        }
        
        let windowSize = Int(vadModelConfig.silero_vad.window_size)
        
        guard let vad else { return }
        
        for offset in stride(from: 0, to: voices.count, by: windowSize) {
            let end = min(offset + windowSize, voices.count)
            vad.acceptWaveform(samples: [Float](voices[offset ..< end]))
        }
        
        while !vad.isEmpty() {
            vad.pop()
        }
        
        let isSpeechDetected = vad.isDetected()
        DispatchQueue.main.async {
            self.isSpeaking = isSpeechDetected
        }
    }

    private func convert(buffer: AVAudioPCMBuffer, time _: Float64) {
        guard let outputFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: sampleRate, channels: 1, interleaved: false) else {
            return
        }

        guard let converter = AVAudioConverter(from: inputFormat, to: outputFormat) else {
            return
        }

        let inputBlock: AVAudioConverterInputBlock = { _, outStatus in
            outStatus.pointee = AVAudioConverterInputStatus.haveData
            return buffer
        }

        let targetFrameCapacity = AVAudioFrameCount(outputFormat.sampleRate) * buffer.frameLength / AVAudioFrameCount(buffer.format.sampleRate)
        if let convertedBuffer = AVAudioPCMBuffer(pcmFormat: outputFormat, frameCapacity: targetFrameCapacity) {
            var error: NSError?
            let status = converter.convert(to: convertedBuffer, error: &error, withInputFrom: inputBlock)

            switch status {
            case .haveData:
                handle(buffer: convertedBuffer)
            case .error:
                if let error {
                    print(error.localizedDescription)
                }
            case .endOfStream:
                streamingData = false
                print("[AudioEngine]: The end of stream has been reached. No data was returned.")
            case .inputRanDry:
                print("[AudioEngine]: Converter input ran dry.")
            @unknown default:
                if error != nil {
                    streamingData = false
                }
                print("[AudioEngine]: Unknown converter error")
            }
        }
    }
}

extension AudioBuffer {
    func array() -> [Float] {
        return Array(UnsafeBufferPointer(self))
    }
}

extension AVAudioPCMBuffer {
    func array() -> [Float] {
        return audioBufferList.pointee.mBuffers.array()
    }
}

