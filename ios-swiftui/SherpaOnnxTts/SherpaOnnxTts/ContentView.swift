//
//  ContentView.swift
//  SherpaOnnxTts
//
//  Created by fangjun on 2023/11/23.
//
// Text-to-speech with Next-gen Kaldi on iOS without Internet connection

import SwiftUI
import AVFoundation
import UniformTypeIdentifiers

class TtsProgressHandler: ObservableObject {
    @Published var progress: Float = 0.0
    @Published var isGenerating = false
    private var audioEngine: AVAudioEngine?
    private var playerNode: AVAudioPlayerNode?
    private var audioFormat: AVAudioFormat?
    private var sampleRate: Float = 22050
    private var pendingBuffers = 0
    private var shouldStop = false

    func startPlayback(sampleRate: Float) {
        self.sampleRate = sampleRate
        self.pendingBuffers = 0
        self.shouldStop = false

        do {
            let session = AVAudioSession.sharedInstance()
            try session.setCategory(.playback, mode: .default)
            try session.setActive(true)
        } catch {
            print("AVAudioSession error: \(error)")
        }

        let engine = AVAudioEngine()
        let player = AVAudioPlayerNode()
        let format = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: Double(sampleRate),
            channels: 1,
            interleaved: false)

        engine.attach(player)
        engine.connect(player, to: engine.mainMixerNode, format: format)

        do {
            try engine.start()
        } catch {
            print("AVAudioEngine start error: \(error)")
            return
        }

        self.audioEngine = engine
        self.playerNode = player
        self.audioFormat = format

        player.play()

        DispatchQueue.main.async {
            self.isGenerating = true
            self.progress = 0.0
        }
    }

    func appendSamples(_ samples: UnsafePointer<Float>?, count: Int32, progress: Float) {
        guard !shouldStop,
              let playerNode = playerNode,
              let audioFormat = audioFormat,
              let samples = samples,
              count > 0
        else { return }

        let frameCount = AVAudioFrameCount(count)
        guard let buffer = AVAudioPCMBuffer(
            pcmFormat: audioFormat, frameCapacity: frameCount)
        else { return }

        buffer.frameLength = frameCount
        let channelData = buffer.floatChannelData![0]
        memcpy(channelData, samples, Int(count) * MemoryLayout<Float>.size)

        pendingBuffers += 1
        playerNode.scheduleBuffer(buffer) { [weak self] in
            DispatchQueue.main.async {
                self?.pendingBuffers -= 1
            }
        }

        DispatchQueue.main.async {
            self.progress = progress
        }
    }

    /// Returns 0 to stop generation, 1 to continue
    var stopFlag: Int32 {
        return shouldStop ? 0 : 1
    }

    func requestStop() {
        shouldStop = true
    }

    func finishGeneration() {
        // Wait for all scheduled buffers to finish playing, then clean up
        DispatchQueue.global(qos: .background).async { [weak self] in
            while let self = self, self.pendingBuffers > 0 && !self.shouldStop {
                Thread.sleep(forTimeInterval: 0.05)
            }

            if self?.shouldStop == true {
                // Immediate stop requested
                self?.playerNode?.stop()
            } else {
                // Let the last buffers play out
                Thread.sleep(forTimeInterval: 0.3)
            }

            DispatchQueue.main.async {
                self?.playerNode?.stop()
                self?.audioEngine?.stop()
                self?.audioEngine = nil
                self?.playerNode = nil
                self?.audioFormat = nil
                self?.isGenerating = false
                if self?.shouldStop != true {
                    self?.progress = 1.0
                }
            }
        }
    }
}

struct ContentView: View {
    @State private var sid = "0"
    @State private var speed = 1.0
    @State private var text = ""
    @State private var showAlert = false
    @State var filename: URL = NSURL() as URL
    @State var audioPlayer: AVAudioPlayer!

    @State private var lang = "en"
    @State private var numSteps = 5

    @StateObject private var progressHandler = TtsProgressHandler()

    @State private var showSavePicker = false

    private let languages = [
        "en", "ko", "ja", "ar", "bg", "cs", "da", "de", "el", "es",
        "et", "fi", "fr", "hi", "hr", "hu", "id", "it", "lt", "lv",
        "nl", "pl", "pt", "ro", "ru", "sk", "sl", "sv", "tr", "uk", "vi"
    ]

    private var tts = createOfflineTts()

    var body: some View {

        VStack(alignment: .leading) {
            HStack {
                Spacer()
                Text("Next-gen Kaldi: TTS").font(.title)
                Spacer()
            }
            if tts.numSpeakers > 1 {
                HStack {
                    Text("Speaker (1-\(tts.numSpeakers))")
                    Stepper("\(Int(sid)! + 1)", value: Binding(
                        get: { (Int(sid) ?? 0) + 1 },
                        set: { sid = "\($0 - 1)" }
                    ), in: 1...Int(tts.numSpeakers))
                }
            }
            HStack{
                Text("Speed \(String(format: "%.1f", speed))")
                    .padding(.trailing)
                Slider(value: $speed, in: 0.5...2.0, step: 0.1) {
                    Text("Speech speed")
                }
            }

            if tts.isSupertonic {
                HStack {
                    Text("Language")
                    Picker("Language", selection: $lang) {
                        ForEach(languages, id: \.self) { l in
                            Text(l).tag(l)
                        }
                    }
                    .pickerStyle(.menu)

                    Spacer()

                    Text("Steps")
                    Stepper("\(numSteps)", value: $numSteps, in: 1...20)
                }
            }

            if progressHandler.isGenerating {
                VStack(spacing: 4) {
                    ProgressView(value: Double(progressHandler.progress))
                        .progressViewStyle(.linear)
                    Text(String(format: "%.0f%%", progressHandler.progress * 100))
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                .padding(.vertical, 4)
            }

            Text("Please input your text below").padding([.trailing, .top, .bottom])

            TextEditor(text: $text)
                .font(.body)
                .opacity(self.text.isEmpty ? 0.25 : 1)
                .disableAutocorrection(true)
                .border(Color.black)
                .frame(minHeight: 100)

            Spacer()
            HStack {
                Spacer()
                if progressHandler.isGenerating {
                    Button(action: {
                        progressHandler.requestStop()
                    }) {
                        Text("Stop")
                    }
                } else {
                    Button(action: {
                        generate()
                    }) {
                        Text("Generate")
                    }
                    .alert(isPresented: $showAlert) {
                        Alert(
                            title: Text("Empty text"),
                            message: Text(
                                "Please input your text before clicking the Generate button"))
                    }
                }
                Spacer()
                Button(action: {
                    self.audioPlayer.play()
                }) {
                    Text("Play")
                }.disabled(filename.absoluteString.isEmpty || progressHandler.isGenerating)
                Spacer()
                Button(action: {
                    showSavePicker = true
                }) {
                    Text("Save")
                }
                .disabled(filename.absoluteString.isEmpty || progressHandler.isGenerating)
                .fileExporter(
                    isPresented: $showSavePicker,
                    document: WavDocument(url: filename),
                    contentType: .wav,
                    defaultFilename: "sherpa-onnx-tts-output"
                ) { result in
                    // fileExporter handles the save
                }
                Spacer()
                Button(action: {
                    shareWav()
                }) {
                    Text("Share")
                }.disabled(filename.absoluteString.isEmpty || progressHandler.isGenerating)
                Spacer()
            }
            Spacer()
        }
        .padding()
    }

    private func generate() {
        let speakerId = Int(self.sid) ?? 0
        let t = self.text.trimmingCharacters(in: .whitespacesAndNewlines)
        if t.isEmpty {
            self.showAlert = true
            return
        }

        if self.filename.absoluteString.isEmpty {
            let tempDirectoryURL = NSURL.fileURL(
                withPath: NSTemporaryDirectory(), isDirectory: true)
            self.filename = tempDirectoryURL.appendingPathComponent("test.wav")
        }

        let handler = progressHandler
        let sampleRate = Float(tts.sampleRate)

        DispatchQueue.global(qos: .userInitiated).async {
            handler.startPlayback(sampleRate: sampleRate)

            let arg = Unmanaged.passUnretained(handler).toOpaque()

            let audio: SherpaOnnxGeneratedAudioWrapper

            if tts.isSupertonic {
                let progressCallback: TtsProgressCallbackWithArg = {
                    samples, n, progress, arg in
                    let h = Unmanaged<TtsProgressHandler>.fromOpaque(arg!)
                        .takeUnretainedValue()
                    h.appendSamples(samples, count: n, progress: progress)
                    return h.stopFlag
                }

                var genConfig = SherpaOnnxGenerationConfigSwift()
                genConfig.sid = speakerId
                genConfig.speed = Float(self.speed)
                genConfig.numSteps = self.numSteps
                genConfig.extra = ["lang": self.lang]
                audio = tts.generateWithConfig(
                    text: t, config: genConfig,
                    callback: progressCallback, arg: arg)
            } else {
                let simpleCallback: TtsCallbackWithArg = { samples, n, arg in
                    let h = Unmanaged<TtsProgressHandler>.fromOpaque(arg!)
                        .takeUnretainedValue()
                    h.appendSamples(samples, count: n, progress: 1.0)
                    return h.stopFlag
                }

                audio = tts.generateWithCallbackWithArg(
                    text: t, callback: simpleCallback, arg: arg,
                    sid: speakerId, speed: Float(self.speed))
            }

            let _ = audio.save(filename: self.filename.path)

            handler.finishGeneration()

            DispatchQueue.main.async {
                self.audioPlayer = try? AVAudioPlayer(contentsOf: self.filename)
            }
        }
    }

    private func shareWav() {
        guard !filename.absoluteString.isEmpty else { return }
        let activityVC = UIActivityViewController(
            activityItems: [filename], applicationActivities: nil)
        guard let scene = UIApplication.shared.connectedScenes.first as? UIWindowScene,
              let rootVC = scene.windows.first?.rootViewController
        else { return }
        rootVC.present(activityVC, animated: true)
    }
}

struct WavDocument: FileDocument {
    static var readableContentTypes: [UTType] { [.wav] }

    let url: URL

    init(url: URL) {
        self.url = url
    }

    init(configuration: ReadConfiguration) throws {
        fatalError("init(configuration:) not implemented")
    }

    func fileWrapper(configuration: WriteConfiguration) throws -> FileWrapper {
        let data = try Data(contentsOf: url)
        return FileWrapper(regularFileWithContents: data)
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
