//
//  ContentView.swift
//  SherpaOnnxSubtitle
//
//  Created by knight on 2023/9/23.
//

import AVKit
import MediaPlayer
import PhotosUI
import SwiftUI

struct ContentView: View {
    @StateObject var subtitleViewModel = SubtitleViewModel()

    var body: some View {
        VStack {
            VStack {
                Text("SherpaOnnxSubtitle")
                    .font(.title)
                VStack(alignment: .leading) {
                    Text("Audio format should be **mono** channel and **16khz** sample rate")

                    Text("You can convert file with the help of ffmpeg")
                    Text("```ffmpeg -i ./foo.mov -acodec pcm_s16le -ac 1 -ar 16000 foo.wav```")
                }
            }
            .padding(.vertical)
            PhotosPicker(
                selection: $subtitleViewModel.selectedItem,
                matching: .videos
            ) {
                Label("Open Audio from Photo Library", systemImage: "photo")
                    .frame(minWidth: 0, maxWidth: .infinity)
                    .padding()
                    .background(.blue, in: .rect(cornerRadius: 8.0))
                    .foregroundColor(.white)
            }

            Button(action: {
                subtitleViewModel.importNow = true
            }, label: {
                Text("Open Audio from Files")
                    .frame(minWidth: 0, maxWidth: .infinity)
                    .padding()
                    .background(.blue, in: .rect(cornerRadius: 8.0))
            })
            .foregroundColor(.white)
            switch subtitleViewModel.loadState {
            case .initial, .loaded(_), .done:
                EmptyView()
            case .loading:
                ProgressView()
            case .failed:
                Text("Gen SRT failed")
            }
        }
        .fileImporter(isPresented: $subtitleViewModel.importNow, allowedContentTypes: [.movie, .audio], onCompletion: handleImportCompletion)
        .onChange(of: subtitleViewModel.importNow) { importNow in
            if !importNow {
                subtitleViewModel.restoreState()
            }
        }
        .fileExporter(isPresented: $subtitleViewModel.exportNow,
                      document: subtitleViewModel.srtDocument, contentType: .srt,
                      defaultFilename: subtitleViewModel.srtName,
                      onCompletion: handleExportCompletion)
        .task(id: subtitleViewModel.selectedItem) {
            do {
                if !subtitleViewModel.hasAudio {
                    return
                }
                subtitleViewModel.loadState = .loading

                if let movie = try await subtitleViewModel.selectedItem?.loadTransferable(type: Audio.self) {
                    subtitleViewModel.loadState = .loaded(movie)
                    subtitleViewModel.generateSRT(from: movie.url)
                } else {
                    subtitleViewModel.loadState = .failed
                }
            } catch {
                subtitleViewModel.loadState = .failed
            }
        }
        .padding()
    }

    private func handleImportCompletion(result: Result<URL, Error>) {
        print("file import...")
        switch result {
        case let .success(file):
            let accessing = file.startAccessingSecurityScopedResource()
            defer {
                if accessing {
                    file.stopAccessingSecurityScopedResource()
                }
            }
            subtitleViewModel.generateSRT(from: file)
        case let .failure(error):
            print(error.localizedDescription)
            subtitleViewModel.loadState = .failed
        }
    }

    private func handleExportCompletion(result: Result<URL, any Error>) {
        switch result {
        case let .success(url):
            print("audio export to: \(url)")
            subtitleViewModel.loadState = .done
        case let .failure(error):
            print("export audio error: \(error.localizedDescription)")
            subtitleViewModel.loadState = .failed
        }
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
