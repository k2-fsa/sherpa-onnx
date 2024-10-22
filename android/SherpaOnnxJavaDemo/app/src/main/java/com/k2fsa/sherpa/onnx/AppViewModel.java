package com.k2fsa.sherpa.onnx;

import androidx.lifecycle.LiveData;
import androidx.lifecycle.MutableLiveData;
import androidx.lifecycle.ViewModel;

public class AppViewModel extends ViewModel {
    private final MutableLiveData<String> speechRecognitionResult = new MutableLiveData<>();

    public LiveData<String> getSpeechRecognitionResult() {
        return speechRecognitionResult;
    }

    public void setSpeechRecognitionResult(String result) {
        speechRecognitionResult.postValue(result);
    }

}
