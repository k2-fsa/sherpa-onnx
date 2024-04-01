//
//  ContentView.swift
//  SherpaOnnxLangID
//
//  Created by knight on 2024/4/1.
//

import SwiftUI

struct ContentView: View {
    @StateObject var viewModel = ViewModel()

    var body: some View {
        VStack {
            Text("ASR with Next-gen Kaldi")
                .font(.title)
            if viewModel.status == .stop {
                Text("See https://github.com/k2-fsa/sherpa-onnx")
                Text("Press the Start button to run!")
            }
            if viewModel.status == .recording {
                Text("Stop will show recording language.")
            }
            Spacer()
            Text("Recording language is: \(viewModel.language)")
                .frame(maxWidth: .infinity)
            Spacer()
            Button {
                toggleRecorder()
            } label: {
                Text(viewModel.status == .stop ? "Start" : "Stop")
            }
        }
        .padding()
    }

    private func toggleRecorder() {
        Task {
            await viewModel.toggleRecorder()
        }
    }
}

#Preview {
    ContentView()
}
