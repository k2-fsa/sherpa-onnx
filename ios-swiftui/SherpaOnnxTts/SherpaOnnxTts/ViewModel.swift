//
//  ViewModel.swift
//  SherpaOnnxTts
//
//  Created by fangjun on 2023/11/23.
//

import Foundation

func createOfflineTts() -> SherpaOnnxOfflineTtsWrapper{
    let vits = sherpaOnnxOfflineTtsVitsModelConfig(model: "", lexicon: "", tokens: "")
    let modelConfig = sherpaOnnxOfflineTtsModelConfig(vits: vits)
    var config = sherpaOnnxOfflineTtsConfig(model: modelConfig)
    return SherpaOnnxOfflineTtsWrapper(config: &config)
}
