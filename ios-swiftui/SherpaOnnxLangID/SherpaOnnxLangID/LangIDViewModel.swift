//
//  LangIDViewModel.swift
//  SherpaOnnxLangID
//
//  Created by knight on 2024/4/1.
//

import SwiftUI
import AVFoundation


enum Status {
    case stop
    case recording
}

@MainActor
class LangIDViewModel: ObservableObject {
    @Published var status: Status = .stop
    @Published var language: String = ""
    
    var voices: [Float] = []
    
    var audioEngine: AVAudioEngine? = nil
    
    var langIdentifier :SherpaOnnxSpokenLanguageIdentificationWrapper? = nil
    
    init() {
        self.initRecorder()
        self.initWhisper()
    }
    
    private func initWhisper() {

        var config = getLangIdentificationTiny()
        self.langIdentifier = SherpaOnnxSpokenLanguageIdentificationWrapper(config: &config)
    }
    
    private func initRecorder() {
        audioEngine = AVAudioEngine()
        let inputNode = self.audioEngine?.inputNode
        let bus = 0
        let inputFormat = inputNode?.outputFormat(forBus: bus)
        let outputFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: 16000, channels: 1,
            interleaved: false)!

        let converter = AVAudioConverter(from: inputFormat!, to: outputFormat)!

        inputNode!.installTap(
            onBus: bus,
            bufferSize: 1024,
            format: inputFormat
        ) {
            (buffer: AVAudioPCMBuffer, when: AVAudioTime) in
            var newBufferAvailable = true

            let inputCallback: AVAudioConverterInputBlock = {
                inNumPackets, outStatus in
                if newBufferAvailable {
                    outStatus.pointee = .haveData
                    newBufferAvailable = false

                    return buffer
                } else {
                    outStatus.pointee = .noDataNow
                    return nil
                }
            }

            let convertedBuffer = AVAudioPCMBuffer(
                pcmFormat: outputFormat,
                frameCapacity:
                    AVAudioFrameCount(outputFormat.sampleRate)
                * buffer.frameLength
                / AVAudioFrameCount(buffer.format.sampleRate))!

            var error: NSError?
            let _ = converter.convert(
                to: convertedBuffer,
                error: &error, withInputFrom: inputCallback)

            let array = convertedBuffer.array()
            if !array.isEmpty {
                self.voices.append(contentsOf: array)
            }
        }
    }

    
    public func toggleRecorder() async {
        if status == .stop {
            await startRecorder()
            status = .recording
        } else {
            await stopRecorder()
            status = .stop
        }
    }
    
    private func startRecorder() async {
        voices = []
        await MainActor.run {
            language = ""
        }

        do {
            try self.audioEngine?.start()
        } catch let error as NSError {
            print("Got an error starting audioEngine: \(error.domain), \(error)")
        }
        print("started")
    }
    
    private func stopRecorder() async {
        audioEngine?.stop()
        print("stopped, and start to identify lang.")
        await identify()
    }
    
    private func identify() async {
        let result = self.langIdentifier?.decode(samples: self.voices)
        if let lang = result?.lang {
            await MainActor.run {
                self.language = lang
                print("voice lang: \(self.language)")
            }
        }
    }
}
