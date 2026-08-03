//
//  SherpaOnnxApp.swift
//  SherpaOnnx
//
//  Created by fangjun on 2023/4/5.
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
struct SherpaOnnxApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}
