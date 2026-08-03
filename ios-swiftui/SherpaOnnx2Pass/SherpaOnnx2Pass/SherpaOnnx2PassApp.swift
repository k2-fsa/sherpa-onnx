//
//  SherpaOnnx2PassApp.swift
//  SherpaOnnx2Pass
//
//  Created by fangjun on 2023/9/11.
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
struct SherpaOnnx2PassApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}
