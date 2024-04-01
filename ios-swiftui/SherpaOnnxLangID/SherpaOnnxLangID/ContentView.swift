//
//  ContentView.swift
//  SherpaOnnxLangID
//
//  Created by knight on 2024/4/1.
//

import SwiftUI

struct ContentView: View {
    @StateObject var viewModel = LangIDViewModel()
    
    var body: some View {
        VStack {
            Text("ASR with Next-gen Kaldi")
                .font(.title)
            if viewModel.status == .stop {
                Text("See https://github.com/k2-fsa/sherpa-onnx")
                Text("Press the Start button to run!")
            }
            if viewModel.status == .recording {
                Text("Recording now, Stop will identify language")
            }
            Spacer()
            Text("Last recording language: \(viewModel.language)")
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
        Task{
            await viewModel.toggleRecorder()
        }
    }
}

#Preview {
    ContentView()
}
