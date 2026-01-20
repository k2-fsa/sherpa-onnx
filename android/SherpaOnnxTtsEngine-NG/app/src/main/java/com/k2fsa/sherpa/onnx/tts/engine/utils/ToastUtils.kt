@file:Suppress("unused")
/* https://github.com/gedoor/legado/blob/master/app/src/main/java/io/legado/app/utils/ToastUtils.kt */
package com.k2fsa.sherpa.onnx.tts.engine.utils

import android.content.Context
import android.widget.Toast
import androidx.annotation.StringRes
import androidx.fragment.app.Fragment

fun Context.toast(@StringRes message: Int, vararg args: Any) {
    runOnUI {
        kotlin.runCatching {
            Toast.makeText(this, getString(message, *args), Toast.LENGTH_SHORT).show()
        }
    }
}

fun Context.toast(message: CharSequence?) {
    runOnUI {
        kotlin.runCatching {
            Toast.makeText(this, message, Toast.LENGTH_SHORT).show()
        }
    }
}

fun Context.longToast(@StringRes message: Int, vararg args: Any) {
    runOnUI {
        kotlin.runCatching {
            Toast.makeText(this, getString(message, *args), Toast.LENGTH_LONG).show()
        }
    }
}

fun Context.longToast(message: CharSequence?) {
    runOnUI {
        kotlin.runCatching {
            Toast.makeText(this, message, Toast.LENGTH_LONG).show()
        }
    }
}


fun Fragment.toast(@StringRes message: Int) = requireActivity().toast(message)

fun Fragment.toast(message: CharSequence) = requireActivity().toast(message)

fun Fragment.longToast(@StringRes message: Int) = requireContext().longToast(message)

fun Fragment.longToast(message: CharSequence) = requireContext().longToast(message)