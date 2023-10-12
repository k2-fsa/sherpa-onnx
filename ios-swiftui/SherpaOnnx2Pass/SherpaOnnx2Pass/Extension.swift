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

extension TimeInterval {
  var hourMinuteSecondMS: String {
    String(format: "%d:%02d:%02d,%03d", hour, minute, second, millisecond)
  }

  var hour: Int {
    Int((self / 3600).truncatingRemainder(dividingBy: 3600))
  }
  var minute: Int {
    Int((self / 60).truncatingRemainder(dividingBy: 60))
  }
  var second: Int {
    Int(truncatingRemainder(dividingBy: 60))
  }
  var millisecond: Int {
    Int((self * 1000).truncatingRemainder(dividingBy: 1000))
  }
}

extension String {
  var fileURL: URL {
    return URL(fileURLWithPath: self)
  }
  var pathExtension: String {
    return fileURL.pathExtension
  }
  var lastPathComponent: String {
    return fileURL.lastPathComponent
  }
  var stringByDeletingPathExtension: String {
    return fileURL.deletingPathExtension().path
  }
}
