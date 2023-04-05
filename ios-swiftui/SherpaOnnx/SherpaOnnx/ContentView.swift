//
//  ContentView.swift
//  SherpaOnnx
//
//  Created by fangjun on 2023/4/5.
//

import SwiftUI

struct ContentView: View {
    @StateObject var sherpaOnnxVM = SherpaOnnxViewModel()

    var body: some View {
        VStack {
            Text("ASR with Next-gen Kaldi")
                .font(.title)
            if sherpaOnnxVM.status == .stop {
                Text("See https://github.com/k2-fsa/sherpa-onnx")
                Text("Press the Start button to run!")
            }
            ScrollView(.vertical, showsIndicators: true) {
                HStack {
                    Text(sherpaOnnxVM.subtitles)
                    Spacer()
                }
            }
            Spacer()
            Button {
                toggleRecorder()
            } label: {
                Text(sherpaOnnxVM.status == .stop ? "Start" : "Stop")
            }
        }
        .padding()
    }

    private func toggleRecorder() {
        sherpaOnnxVM.toggleRecorder()
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
