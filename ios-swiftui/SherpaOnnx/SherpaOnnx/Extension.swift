//
//  Extension.swift
//  SherpaOnnx
//
//  Created by knight on 2023/4/5.
//

import AVFoundation

extension AudioBuffer {
    func array() -> [Float] {
        return Array(UnsafeBufferPointer(self))
    }
}

extension AVAudioPCMBuffer {
    func array() -> [Float] {
        return self.audioBufferList.pointee.mBuffers.array()
    }
}
