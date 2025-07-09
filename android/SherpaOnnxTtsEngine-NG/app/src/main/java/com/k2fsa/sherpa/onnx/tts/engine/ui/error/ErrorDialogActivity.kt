package com.k2fsa.sherpa.onnx.tts.engine.ui.error

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.graphics.Typeface
import android.os.Bundle
import android.text.Spannable
import android.text.SpannableStringBuilder
import android.text.style.StyleSpan
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Error
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import com.k2fsa.sherpa.onnx.tts.engine.AppConst
import com.k2fsa.sherpa.onnx.tts.engine.R
import com.k2fsa.sherpa.onnx.tts.engine.ui.theme.SherpaOnnxTtsEngineTheme
import com.k2fsa.sherpa.onnx.tts.engine.ui.widgets.AppDialog
import com.k2fsa.sherpa.onnx.tts.engine.ui.widgets.LoadingContent
import com.k2fsa.sherpa.onnx.tts.engine.ui.widgets.OkButton
import com.k2fsa.sherpa.onnx.tts.engine.utils.ClipboardUtils
import com.k2fsa.sherpa.onnx.tts.engine.utils.runOnUI
import com.k2fsa.sherpa.onnx.tts.engine.utils.toast
import java.util.UUID

fun Context.showErrorDialog(t: Throwable, title: String? = null) {
    runOnUI {
        ErrorDialogActivity.start(this, title ?: "Error", t)
    }
}

@Suppress("DEPRECATION")
class ErrorDialogActivity : ComponentActivity() {
    companion object {
        const val ACTION_FINISH =
            "com.k2fsa.sherpa.onnx.tts.engine.ui.error.ErrorDialogActivity.ACTION_FINISH"

        const val KEY_T_DATA = "throwable"
        const val KEY_TITLE = "title"
        private const val KEY_ID = "id"

        val vm by lazy { ErrorDialogViewModel() }

        fun start(context: Context, title: String, t: Throwable) {
            val id = UUID.randomUUID().toString()
            vm.throwableList[id] = t
            context.startActivity(Intent(context, ErrorDialogActivity::class.java).apply {
                putExtra(KEY_TITLE, title)
                putExtra(KEY_ID, id)
            })
        }
    }

    private val mReceiver by lazy { MyReceiver() }

    inner class MyReceiver : BroadcastReceiver() {
        override fun onReceive(context: Context?, intent: Intent?) {
            if (intent?.action == ACTION_FINISH) {
                finish()
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        val id = intent.getStringExtra(KEY_ID) ?: return
        vm.throwableList.remove(id)

        AppConst.localBroadcast.unregisterReceiver(mReceiver)
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val lp = window.attributes
        lp.alpha = 0.0f
        window.attributes = lp

        AppConst.localBroadcast.registerReceiver(mReceiver, IntentFilter(ACTION_FINISH))

        val title = intent.getStringExtra(KEY_TITLE) ?: getString(R.string.error)
        val t = vm.throwableList[intent.getStringExtra(KEY_ID) ?: ""]
            ?: intent.getSerializableExtra(KEY_T_DATA) as? Throwable

        if (t == null) {
            toast(R.string.error)
            finish()
            return
        }

        val str = t.stackTraceToString()
        setContent {
            SherpaOnnxTtsEngineTheme {
                var showDialog by remember { mutableStateOf(true) }
                var isLoading by remember { mutableStateOf(false) }
                AppDialog(
                    onDismissRequest = {
                        showDialog = false
                        finish()
                    },
                    title = {
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Icon(
                                Icons.Default.Error,
                                contentDescription = null,
                                tint = MaterialTheme.colorScheme.error
                            )
                            Spacer(modifier = Modifier.width(8.dp))
                            Text(title)
                        }
                    },
                    content = {
                        LoadingContent(isLoading = isLoading) {
                            Column {
                                SelectionContainer {
                                    Text(
                                        text = t.localizedMessage ?: "",
                                        style = MaterialTheme.typography.bodyMedium,
                                        color = MaterialTheme.colorScheme.error
                                    )
                                }
                                ThrowableText(t = str)
                            }
                        }
                    },
                    buttons = {
                        Row(Modifier.fillMaxWidth()) {
                            TextButton(modifier = Modifier.padding(end = 8.dp), onClick = {
                                ClipboardUtils.copyText(str)
                                toast(R.string.copied)
                                showDialog = false
                                finish()
                            }) {
                                Text(stringResource(id = android.R.string.copy))
                            }

                            Row(Modifier.weight(1f)) {
                                Spacer(modifier = Modifier.weight(1f))

//                                TextButton(enabled = !isLoading,
//                                    onClick = {
//
//                                        lifecycleScope.launch(Dispatchers.Main) {
//                                            isLoading = true
//                                            kotlin.runCatching {
//                                                val url = withIO { Tts_server_lib.uploadLog(str) }
//                                                ClipboardUtils.copyText(url)
//                                                longToast(R.string.copied)
//                                            }.onFailure {
//                                                longToast(
//                                                    getString(
//                                                        R.string.upload_failed,
//                                                        it.message
//                                                    )
//                                                )
//                                            }
//                                            isLoading = false
//                                        }
//                                    }) {
//                                    Text(stringResource(id = R.string.upload_to_url))
//                                }

                                OkButton {
                                    showDialog = false
                                    finish()
                                }
                            }
                        }
                    }
                )
            }
        }
    }

    @Composable
    fun ThrowableText(modifier: Modifier = Modifier, t: String) {
        AndroidView(
            modifier = modifier,
            factory = { ctx ->
                val tv = BigTextView(ctx)

                t.lines().forEach {
                    val span = if (it.trimStart().startsWith("at")) {
                        SpannableStringBuilder(it).apply {
                            setSpan(
                                StyleSpan(Typeface.ITALIC),
                                0,
                                it.length,
                                Spannable.SPAN_EXCLUSIVE_EXCLUSIVE
                            )
                        }
                    } else {
                        SpannableStringBuilder(it).apply {
                            setSpan(
                                StyleSpan(Typeface.BOLD),
                                0,
                                it.length,
                                Spannable.SPAN_EXCLUSIVE_EXCLUSIVE
                            )
                        }
                    }

                    tv.append(span)
                    tv.append("\n")
                }

                tv
            }
        )
    }
}