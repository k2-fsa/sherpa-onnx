//
//  ContentView.swift
//  SherpaOnnxTts
//
//  Created by fangjun on 2023/11/23.
//

import SwiftUI

struct ContentView: View {
    @State private var sid = "0"
    @State private var speed = 1.0
    @State private var text = ""
    var body: some View {
        
        VStack(alignment: .leading) {
            HStack {
                Spacer()
                Text("Next-gen Kaldi: TTS").font(.title)
                Spacer()
            }
            HStack{
                Text("Speaker ID")
                TextField("Please input a speaker ID", text: $sid).textFieldStyle(.roundedBorder)
                    .keyboardType(.numberPad)
            }
            HStack{
                Text("Speed \(String(format: "%.1f", speed))")
                    .padding(.trailing)
                Slider(value: $speed, in: 0.5...2.0, step: 0.1) {
                    Text("Speech speed")
                }
            }
            
            Text("Please input your text below").padding([.trailing, .top, .bottom])
            
            TextEditor(text: $text)
                .font(.body)
                .opacity(self.text.isEmpty ? 0.25 : 1)
                .disableAutocorrection(true)
                .border(Color.black)
            
            Spacer()
            HStack {
                Spacer()
                Button(action: {
                    text += "1"
                }) {
                    Text("Generate")
                }
                Spacer()
                Button(action:{}){
                    Text("Play")
                }
                Spacer()
            }
            Spacer()
        }
        .padding()
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
