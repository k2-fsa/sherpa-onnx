//
//  SherpaOnnxSubtitleApp.swift
//  SherpaOnnxSubtitle
//
//  Created by knight on 2023/9/23.
//

#if canImport(SherpaOnnx)
import SherpaOnnx
#elseif canImport(SherpaOnnxShared)
import SherpaOnnxShared
#else
#error("SherpaOnnx module not found. Please check your SPM dependency configuration.")
#endif
import SwiftUI

@main
struct SherpaOnnxSubtitleApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}
