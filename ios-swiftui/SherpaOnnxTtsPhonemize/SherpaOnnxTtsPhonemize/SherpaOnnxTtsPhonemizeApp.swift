//
//  SherpaOnnxTtsPhonemizeApp.swift
//  SherpaOnnxTtsPhonemize
//
//  Created by fangjun on 2023/11/23.
//

import SwiftUI

@main
struct SherpaOnnxTtsPhonemizeApp: App {
    init() {
        // The phonemizer is only used by models with a .phonemize frontend.
        initPiperPhonemize()
    }

    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}
