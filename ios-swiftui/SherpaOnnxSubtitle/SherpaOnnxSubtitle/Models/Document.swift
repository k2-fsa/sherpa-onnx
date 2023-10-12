//
//  Document.swift
//  YPlayer
//
//  Created by knight on 2023/6/5.
//

import SwiftUI
import UniformTypeIdentifiers

struct Document: FileDocument {
    static var readableContentTypes = [UTType.srt]
    static var writableContentTypes = [UTType.srt]
    var data: Data?

    init(data: Data?) {
        self.data = data
    }

    init(configuration: ReadConfiguration) throws {
        if let data = configuration.file.regularFileContents {
            self.data = data
        }
    }

    func fileWrapper(configuration _: WriteConfiguration) throws -> FileWrapper {
        guard let data = data else {
            throw ExportError.fileNotFound
        }
        return FileWrapper(regularFileWithContents: data)
    }
}
