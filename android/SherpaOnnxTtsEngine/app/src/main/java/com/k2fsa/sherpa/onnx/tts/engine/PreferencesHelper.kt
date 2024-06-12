import android.content.Context
import android.content.SharedPreferences

class PreferenceHelper(context: Context) {

    private val PREFS_NAME = "com.k2fsa.sherpa.onnx.tts.engine"
    private val SPEED_KEY = "speed"
    private val SID_KEY = "speaker_id"

    private val sharedPreferences: SharedPreferences =
        context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)

    fun setSpeed(value: Float) {
        val editor = sharedPreferences.edit()
        editor.putFloat(SPEED_KEY, value)
        editor.apply()
    }

    fun getSpeed(): Float {
        return sharedPreferences.getFloat(SPEED_KEY, 1.0f)
    }

    fun setSid(value: Int) {
        val editor = sharedPreferences.edit()
        editor.putInt(SID_KEY, value)
        editor.apply()
    }

    fun getSid(): Int {
        return sharedPreferences.getInt(SID_KEY, 0)
    }
}