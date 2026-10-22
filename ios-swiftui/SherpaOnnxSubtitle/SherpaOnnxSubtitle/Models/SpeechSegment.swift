//
//  SpeechSegment.swift
//  SherpaOnnxSubtitle
//
//  Created by knight on 2023/9/23.
//

import Foundation

class SpeechSegment: CustomStringConvertible {
    let start: Double
    let end: Double
    let text: String

    init(start: Double, duration: Double, text: String) {
        self.start = start
        end = start + duration
        self.text = text
    }

    public var description: String {
        var s: String
        s = TimeInterval(start).hourMinuteSecondMS
        s += " --> "
        s += TimeInterval(end).hourMinuteSecondMS
        s += "\n"
        s += text

        return s
    }
}
