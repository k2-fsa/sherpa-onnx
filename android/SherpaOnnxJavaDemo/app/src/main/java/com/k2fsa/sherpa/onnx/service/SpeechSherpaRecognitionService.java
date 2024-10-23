package com.k2fsa.sherpa.onnx.service;

import android.Manifest;
import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.Service;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.content.res.AssetManager;
import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.MediaRecorder;
import android.os.Build;
import android.os.IBinder;
import android.text.TextUtils;
import android.util.Log;

import androidx.core.app.ActivityCompat;
import androidx.core.app.NotificationCompat;


import com.k2fsa.sherpa.onnx.AppViewModel;
import com.k2fsa.sherpa.onnx.Application;

import com.k2fsa.sherpa.onnx.OnlineModelConfig;
import com.k2fsa.sherpa.onnx.OnlineRecognizer;

import com.k2fsa.sherpa.onnx.OnlineRecognizerConfig;
import com.k2fsa.sherpa.onnx.OnlineStream;
import com.k2fsa.sherpa.onnx.OnlineTransducerModelConfig;
import com.k2fsa.sherpa.onnx.R;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

import java.util.Objects;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;


public class SpeechSherpaRecognitionService extends Service {

    private AppViewModel appViewModel;
    private OnlineRecognizer recognizer;
    private final int sampleRateInHz = 16000;

    private Thread recordingThread;
    private boolean isRecording = false;
    private int audioSource = MediaRecorder.AudioSource.MIC;
    private int channelConfig = AudioFormat.CHANNEL_IN_MONO;
    private int audioFormat = AudioFormat.ENCODING_PCM_16BIT;
    private AudioRecord audioRecord;
    private int idx = 0;
    private String lastText = "";
    private ExecutorService executor;

    @Override
    public void onCreate() {
        super.onCreate();
        startForegroundService();
        // 获取 ViewModel
        appViewModel = Application.getInstance().getViewModel();
        int numBytes = AudioRecord.getMinBufferSize(sampleRateInHz, channelConfig, audioFormat);

        audioRecord = new AudioRecord(
                audioSource,
                sampleRateInHz,
                channelConfig,
                audioFormat,
                numBytes * 2 // a sample has two bytes as we are using 16-bit PCM
        );
        executor = Executors.newSingleThreadExecutor();
        executor.execute(this::initializeSherpa);
    }


    private void initializeSherpa() {
        Log.d("Current Directory", System.getProperty("user.dir"));
        String modelDir = "sherpa-onnx-streaming-zipformer-zh-14M-2023-02-23";
        initializeSherpaDir(modelDir, modelDir);
        OnlineTransducerModelConfig onlineTransducerModelConfig = OnlineTransducerModelConfig.builder()
                .setEncoder(modelDir + "/encoder-epoch-99-avg-1.int8.onnx")
                .setDecoder(modelDir + "/decoder-epoch-99-avg-1.onnx")
                .setJoiner(modelDir + "/joiner-epoch-99-avg-1.int8.onnx")
                .build();

        OnlineModelConfig onlineModelConfig = OnlineModelConfig.builder()
                .setTransducer(onlineTransducerModelConfig)
                .setTokens(modelDir + "/tokens.txt")
                .setModelType("zipformer")
                .build();
        OnlineRecognizerConfig config = OnlineRecognizerConfig.builder()
                .setOnlineModelConfig(onlineModelConfig)
                .build();
        recognizer = new OnlineRecognizer(getAssets(), config);

        audioRecord.startRecording();
        startRecognition();
    }

    private void startRecognition() {
        isRecording = true;
        recordingThread = new Thread(this::processSamples);
        recordingThread.start();
    }

    private void processSamples() {
        OnlineStream stream = recognizer.createStream();
        double interval = 0.1;
        int bufferSize = (int) (interval * sampleRateInHz);
        short[] buffer = new short[bufferSize];

        while (isRecording) {
            int ret = audioRecord != null ? audioRecord.read(buffer, 0, buffer.length) : -1;
            if (ret > 0) {
                float[] samples = new float[ret];
                for (int i = 0; i < ret; i++) {
                    samples[i] = buffer[i] / 32768.0f;
                }
                stream.acceptWaveform(samples, sampleRateInHz);
                while (recognizer.isReady(stream)) {
                    recognizer.decode(stream);
                }

                boolean isEndpoint = recognizer.isEndpoint(stream);
                String text = recognizer.getResult(stream).getText();
                if (isEndpoint) {
                    float[] tailPaddings = new float[(int) (0.8 * sampleRateInHz)];
                    stream.acceptWaveform(tailPaddings, sampleRateInHz);
                    while (recognizer.isReady(stream)) {
                        recognizer.decode(stream);
                    }
                    text = recognizer.getResult(stream).getText();
                }

                String textToDisplay = lastText;

                if (!TextUtils.isEmpty(text)) {
                    textToDisplay = TextUtils.isEmpty(text) ? idx + ": " + text : lastText + "\n" + idx + ": " + text;
                }

                if (isEndpoint) {
                    recognizer.reset(stream);
                    if (!TextUtils.isEmpty(text)) {
                        lastText = lastText + "\n" + idx + ": " + text;
                        textToDisplay = lastText;
                        idx += 1;
                    }
                    appViewModel.setSpeechRecognitionResult(textToDisplay);
                }
            }

        }
        stream.release();

    }


    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {

        return START_STICKY;
    }

    @Override
    public void onDestroy() {
        super.onDestroy();
        audioRecord.stop();
        audioRecord.release();
        executor.shutdown();
        stopForeground(true);
    }

    @Override
    public IBinder onBind(Intent intent) {
        return null;
    }


    private void startForegroundService() {
        String channelId = createNotificationChannel();

        Notification notification = new NotificationCompat.Builder(this, channelId)
                .setContentTitle("Foreground Service")
                .setContentText("Running in the foreground")
                .setSmallIcon(R.drawable.ic_bg_mic_24)
                .build();

        startForeground(1, notification);
    }

    // 创建通知渠道 (针对 Android 8.0 及以上版本)
    private String createNotificationChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            String channelId = "speech_channel";
            String channelName = "Speech Channel";
            NotificationChannel channel = new NotificationChannel(channelId, channelName, NotificationManager.IMPORTANCE_LOW);
            NotificationManager manager = getSystemService(NotificationManager.class);
            if (manager != null) {
                manager.createNotificationChannel(channel);
            }
            return channelId;
        } else {
            return "";
        }
    }

    private void initializeSherpaDir(String assetDir, String internalDir) {
        AssetManager assetManager = getAssets();
        File outDir = new File(getFilesDir(), internalDir);

        if (!outDir.exists()) {
            outDir.mkdirs();
        }

        try {
            String[] assets = assetManager.list(assetDir);
            if (assets != null) {
                for (String asset : assets) {
                    String assetPath = assetDir.isEmpty() ? asset : assetDir + "/" + asset;
                    File outFile = new File(outDir, asset);
                    if (Objects.requireNonNull(assetManager.list(assetPath)).length > 0) {
                        outFile.mkdirs();
                        initializeSherpaDir(assetPath, internalDir + "/" + asset); // 递归复制子目录
                    } else {
                        InputStream in = assetManager.open(assetPath);
                        OutputStream out = new FileOutputStream(outFile);

                        byte[] buffer = new byte[1024];
                        int read;
                        while ((read = in.read(buffer)) != -1) {
                            out.write(buffer, 0, read);
                        }

                        in.close();
                        out.flush();
                        out.close();
                    }
                }
            }
        } catch (IOException e) {
            Log.e("ModelCopy", "Failed to copy assets", e);
        }
    }
}
