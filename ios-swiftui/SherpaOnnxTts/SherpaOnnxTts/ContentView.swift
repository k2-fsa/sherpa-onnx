//
//  ContentView.swift
//  SherpaOnnxTts
//
//  Created by fangjun on 2023/11/23.
//
// Speech-to-text with Next-gen Kaldi on iOS without Internet connection

import SwiftUI
import AVFoundation

struct ContentView: View {
    @State private var sid = "0"
    @State private var speed = 1.0
    @State private var text = ""
    @State private var showAlert = false
    @State var filename: URL = NSURL() as URL
    @State var audioPlayer: AVAudioPlayer!

    private var tts = createOfflineTts()

    var body: some View {

        VStack(alignment: .leading) {
            HStack {
                Spacer()
                Text("Next-gen Kaldi: TTS").font(.title)
                Spacer()
            }
            HStack{
                Text("Speaker ID")
                TextField("Please input a speaker ID", text: $sid).textFieldStyle(.roundedBorder)
                    .keyboardType(.numberPad)
            }
            HStack{
                Text("Speed \(String(format: "%.1f", speed))")
                    .padding(.trailing)
                Slider(value: $speed, in: 0.5...2.0, step: 0.1) {
                    Text("Speech speed")
                }
            }

            Text("Please input your text below").padding([.trailing, .top, .bottom])

            TextEditor(text: $text)
                .font(.body)
                .opacity(self.text.isEmpty ? 0.25 : 1)
                .disableAutocorrection(true)
                .border(Color.black)

            Spacer()
            HStack {
                Spacer()
                Button(action: {
                    let speakerId = Int(self.sid) ?? 0
                    let t = self.text.trimmingCharacters(in: .whitespacesAndNewlines)
                    if t.isEmpty {
                        self.showAlert = true
                        return
                    }

                    let audio = tts.generate(text: t, sid: speakerId, speed: Float(self.speed))
                    if self.filename.absoluteString.isEmpty {
                        let tempDirectoryURL = NSURL.fileURL(withPath: NSTemporaryDirectory(), isDirectory: true)
                        self.filename = tempDirectoryURL.appendingPathComponent("test.wav")
                    }

                    let _ = audio.save(filename: filename.path)

                    self.audioPlayer = try! AVAudioPlayer(contentsOf: filename)
                    self.audioPlayer.play()
                }) {
                    Text("Generate")
                }.alert(isPresented: $showAlert) {
                    Alert(title: Text("Empty text"), message: Text("Please input your text before clicking the Generate button"))
                }
                Spacer()
                Button (action: {
                    self.audioPlayer.play()
                }) {
                    Text("Play")
                }.disabled(filename.absoluteString.isEmpty)
                Spacer()
            }
            Spacer()
        }
        .padding()
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
