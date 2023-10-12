//
//  ContentView.swift
//  SherpaOnnxVad
//
//  Created by knight on 2023/10/11.
//

import SwiftUI

struct ContentView: View {
    @StateObject var viewModel = ViewModel()
    
    var body: some View {
        VStack {
            Text("Next-gen Kaldi: SileroVAD")
                .font(.title)
            Spacer()
            Circle()
                .foregroundColor(viewModel.indicatorColor)
                .padding(40)
            Spacer()
            Button(action: {
                viewModel.toggle()
            }, label: {
                Text(viewModel.buttonInstruction)
                    .font(.title)
                    .frame(maxWidth: .infinity)
            })
            .buttonStyle(.borderedProminent)
        }
        .padding()
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
