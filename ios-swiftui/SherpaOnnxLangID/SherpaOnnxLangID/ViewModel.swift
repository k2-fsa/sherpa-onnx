//
//  ViewModel.swift
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
class ViewModel:ObservableObject {
    @Published var status: Status = .stop

    @Published var language: String = ""
    
    var languageIdentifier: SherpaOnnxSpokenLanguageIdentificationWrapper? = nil
    var audioEngine: AVAudioEngine? = nil
    
    var voices: [Float] = []

    init() {
        initRecorder()
        initRecognizer()
    }
    
    private func initRecognizer() {
        var config =  getLanguageIdentificationTiny()
        self.languageIdentifier = SherpaOnnxSpokenLanguageIdentificationWrapper(config: &config)
    }
    
    private func initRecorder() {
        print("init recorder")
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

            // TODO(fangjun): Handle status != haveData

            let array = convertedBuffer.array()
            if !array.isEmpty {
                self.voices.append(contentsOf: array)
            }
        }
    }
    
    public func toggleRecorder() async{
        if status == .stop {
            await startRecorder()
        } else {
            await stopRecorder()
        }
    }

    private func startRecorder() async {
        await MainActor.run {
            self.language = ""
        }
        if !self.voices.isEmpty {
            self.voices = []
        }
        do {
            try self.audioEngine?.start()
            status = .recording
            print("started")
        } catch let error as NSError {
            print("Got an error starting audioEngine: \(error.domain), \(error)")
        }
    }

    private func stopRecorder() async {
        audioEngine?.stop()
        print("stopped, and begin identify language")
        await self.identify()
        status = .stop
    }
    
    private func identify() async {
        let result = self.languageIdentifier?    .decode(samples: self.voices)
        if let language = result?.lang {
            await MainActor.run {
                self.language = language
            }
        }
    }
}
