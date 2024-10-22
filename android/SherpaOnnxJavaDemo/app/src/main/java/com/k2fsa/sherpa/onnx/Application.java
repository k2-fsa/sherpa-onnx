package com.k2fsa.sherpa.onnx;

import androidx.annotation.NonNull;
import androidx.lifecycle.ViewModelProvider;
import androidx.lifecycle.ViewModelStore;
import androidx.lifecycle.ViewModelStoreOwner;


public class Application extends android.app.Application implements ViewModelStoreOwner {
    public static Application sApplication;


    private AppViewModel viewModel;
    private ViewModelStore viewModelStore;

    public static Application getInstance() {
        return sApplication;
    }

    @Override
    public void onCreate() {
        super.onCreate();
        sApplication = this;
        viewModelStore = new ViewModelStore();
        viewModel = new ViewModelProvider(this).get(AppViewModel.class);
    }

    @NonNull
    @Override
    public ViewModelStore getViewModelStore() {
        return viewModelStore;
    }

    public AppViewModel getViewModel() {
        return viewModel;
    }


}
