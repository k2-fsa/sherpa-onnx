//
//  SherpaOnnxLangIDApp.swift
//  SherpaOnnxLangID
//
//  Created by knight on 2024/4/1.
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
struct SherpaOnnxLangIDApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}
