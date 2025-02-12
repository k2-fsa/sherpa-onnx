package com.k2fsa.sherpa.onnx.tts.engine.ui.error

import android.content.Context
import android.util.AttributeSet
import android.widget.FrameLayout
import android.widget.TextView
import com.k2fsa.sherpa.onnx.tts.engine.R

class BigTextView(context: Context, attrs: AttributeSet?, defaultStyle: Int) :
    FrameLayout(context, attrs, defaultStyle) {
    constructor(context: Context, attrs: AttributeSet?) : this(context, attrs, 0)
    constructor(context: Context) : this(context, null, 0)

    init {
        inflate(context, R.layout.big_text_view, this)
    }

    private val mText by lazy {
        findViewById<TextView>(R.id.tv_log)
    }

    fun setText(text: CharSequence) {
        mText.text = text
    }

    fun append(text: CharSequence) {
        mText.append(text)
    }
}