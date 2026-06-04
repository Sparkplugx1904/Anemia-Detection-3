package com.anedet.madyapadma.camera

import android.content.Context
import android.content.SharedPreferences
import androidx.core.content.edit
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow

/**
 * Pengaturan aplikasi yang persisten menggunakan SharedPreferences.
 *
 * - smartAutoCapture: aktif/nonaktif fitur auto-capture
 * - confidenceThreshold: ambang deteksi (0.10–0.90)
 * - stabilityFrames: jumlah frame berturut-turut yang harus terdeteksi sebelum auto-capture
 * - sharpnessMin: Laplacian variance minimum (blur detection)
 * - language: kode bahasa saat ini
 */
class AppSettings(context: Context) {

    private val prefs: SharedPreferences =
        context.applicationContext.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)

    private val _smartAutoCapture = MutableStateFlow(prefs.getBoolean(KEY_AUTO, true))
    val smartAutoCapture: StateFlow<Boolean> = _smartAutoCapture.asStateFlow()

    private val _confidenceThreshold = MutableStateFlow(
        prefs.getFloat(KEY_THRESHOLD, DEFAULT_THRESHOLD).coerceIn(0.10f, 0.90f)
    )
    val confidenceThreshold: StateFlow<Float> = _confidenceThreshold.asStateFlow()

    private val _stabilityFrames = MutableStateFlow(prefs.getInt(KEY_STABILITY, DEFAULT_STABILITY))
    val stabilityFrames: StateFlow<Int> = _stabilityFrames.asStateFlow()

    private val _sharpnessMin = MutableStateFlow(prefs.getFloat(KEY_SHARPNESS, DEFAULT_SHARPNESS))
    val sharpnessMin: StateFlow<Float> = _sharpnessMin.asStateFlow()

    private val _language = MutableStateFlow(prefs.getString(KEY_LANGUAGE, DEFAULT_LANGUAGE) ?: DEFAULT_LANGUAGE)
    val language: StateFlow<String> = _language.asStateFlow()

    fun setSmartAutoCapture(enabled: Boolean) {
        _smartAutoCapture.value = enabled
        prefs.edit { putBoolean(KEY_AUTO, enabled) }
    }

    fun setConfidenceThreshold(value: Float) {
        val clamped = value.coerceIn(0.10f, 0.90f)
        _confidenceThreshold.value = clamped
        prefs.edit { putFloat(KEY_THRESHOLD, clamped) }
    }

    fun setStabilityFrames(value: Int) {
        val clamped = value.coerceIn(2, 10)
        _stabilityFrames.value = clamped
        prefs.edit { putInt(KEY_STABILITY, clamped) }
    }

    fun setSharpnessMin(value: Float) {
        val clamped = value.coerceIn(1f, 200f)
        _sharpnessMin.value = clamped
        prefs.edit { putFloat(KEY_SHARPNESS, clamped) }
    }

    fun setLanguage(lang: String) {
        _language.value = lang
        prefs.edit { putString(KEY_LANGUAGE, lang) }
    }

    companion object {
        private const val PREFS_NAME = "anedet_settings"
        private const val KEY_AUTO = "smart_auto_capture"
        private const val KEY_THRESHOLD = "confidence_threshold"
        private const val KEY_STABILITY = "stability_frames"
        private const val KEY_SHARPNESS = "sharpness_min"
        private const val KEY_LANGUAGE = "language"

        const val DEFAULT_THRESHOLD = 0.35f
        const val DEFAULT_STABILITY = 4
        const val DEFAULT_SHARPNESS = 8f
        const val DEFAULT_LANGUAGE = "en"
    }
}
