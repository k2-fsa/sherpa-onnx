package com.k2fsa.sherpa.onnx

import android.os.Bundle
import android.view.View
import android.widget.Button
import androidx.appcompat.app.AppCompatActivity


class MainActivity : AppCompatActivity() {

    private lateinit var recordButton: Button
    private lateinit var circle: View

    @Volatile
    private var isRecording: Boolean = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        circle= findViewById(R.id.powerCircle)

        recordButton = findViewById(R.id.record_button)
        recordButton.setOnClickListener { onclick() }
    }

    private fun onclick() {
        if (!isRecording) {
            isRecording = true
            onVad(true)
        } else {
            isRecording= false
            onVad(false)
        }
    }

    private fun onVad(isSpeech: Boolean) {
        if(isSpeech) {
            circle.background = resources.getDrawable(R.drawable.green_circle)
        } else {
            circle.background = resources.getDrawable(R.drawable.gray_circle)
        }

    }
}