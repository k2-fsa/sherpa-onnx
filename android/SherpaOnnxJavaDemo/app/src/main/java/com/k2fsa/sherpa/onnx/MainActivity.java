package com.k2fsa.sherpa.onnx;

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.content.ContextCompat;
import androidx.lifecycle.ViewModelProvider;

import android.Manifest;
import android.content.Intent;
import android.os.Bundle;
import android.util.Log;
import android.widget.TextView;

import com.k2fsa.sherpa.onnx.service.SpeechSherpaRecognitionService;

import pub.devrel.easypermissions.EasyPermissions;

public class MainActivity extends AppCompatActivity {
    private AppViewModel appViewModel;
    private TextView tvText;
    private static final int RC_AUDIO_PERM = 123;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        tvText = findViewById(R.id.text);
        requestMicrophonePermission();
    }


    private void startSpeechService() {
        Intent serviceIntent = new Intent(this, SpeechSherpaRecognitionService.class);
        ContextCompat.startForegroundService(this, serviceIntent);
        appViewModel = new ViewModelProvider(Application.getInstance()).get(AppViewModel.class);
        appViewModel.getSpeechRecognitionResult().observe(this, this::handleSpeechRecognitionResult);
    }

    private void handleSpeechRecognitionResult(String result) {
        tvText.setText(result);
    }

    private void requestMicrophonePermission() {
        String[] perms = {Manifest.permission.RECORD_AUDIO};
        if (EasyPermissions.hasPermissions(this, perms)) {
            startSpeechService();
        } else {
            EasyPermissions.requestPermissions(MainActivity.this,
                    "We need access to your microphone for voice recognition",
                    RC_AUDIO_PERM, perms);
        }
    }
}