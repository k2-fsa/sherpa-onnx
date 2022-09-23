#include <iostream>

#include "kaldi_native_io/csrc/kaldi-io.h"
#include "kaldi_native_io/csrc/wave-reader.h"
#include "kaldi-native-fbank/csrc/online-feature.h"


kaldiio::Matrix<float> readWav(std::string filename, bool log = false){
    if (log)
        std::cout << "reading " << filename << std::endl;
    
    bool binary = true;
    kaldiio::Input ki(filename, &binary);
    kaldiio::WaveHolder wh;

    if (!wh.Read(ki.Stream())) {
        std::cerr << "Failed to read " << filename;
        exit(EXIT_FAILURE);
    }

    auto &wave_data = wh.Value();
    auto &d = wave_data.Data();

    if (log)
        std::cout << "wav shape: " << "(" << d.NumRows() << "," << d.NumCols() << ")" << std::endl;

    return d;
}


std::vector<float> ComputeFeatures(knf::OnlineFbank &fbank, knf::FbankOptions opts, kaldiio::Matrix<float> samples, bool log = false){
    int numSamples = samples.NumCols();

    for (int i = 0; i < numSamples; i++)
    {
        float currentSample = samples.Row(0).Data()[i] / 32768;
        fbank.AcceptWaveform(opts.frame_opts.samp_freq, &currentSample, 1);
    }
    
    std::vector<float> features;
    int32_t num_frames = fbank.NumFramesReady();
    for (int32_t i = 0; i != num_frames; ++i) {
        const float *frame = fbank.GetFrame(i);
        for (int32_t k = 0; k != opts.mel_opts.num_bins; ++k) {
            features.push_back(frame[k]);
        }
    }
    if (log){
        std::cout << "done feature extraction" << std::endl;
        std::cout << "extracted fbank shape " << "(" << num_frames << "," << opts.mel_opts.num_bins << ")" << std::endl;
        
        for (int i=0; i< 20; i++)
            std::cout << features.at(i) << std::endl;
    }

    return features;
}