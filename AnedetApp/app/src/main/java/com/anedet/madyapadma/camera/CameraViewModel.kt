package com.anedet.madyapadma.camera

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import android.graphics.BitmapFactory
import com.anedet.madyapadma.ml.AnemiaPipeline
import com.anedet.madyapadma.ml.ResultImageSaver
import com.anedet.madyapadma.model.PredictionResult
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

class CameraViewModel(application: Application) : AndroidViewModel(application) {
    private val pipeline = AnemiaPipeline(application)

    private val _predictionResult = MutableStateFlow<PredictionResult?>(null)
    val predictionResult = _predictionResult.asStateFlow()

    private val _isAnalyzing = MutableStateFlow(false)
    val isAnalyzing = _isAnalyzing.asStateFlow()
    private val _saveToDevice = MutableStateFlow(true)
    val saveToDevice = _saveToDevice.asStateFlow()

    private val _isAutoCaptureEnabled = MutableStateFlow(false)
    val isAutoCaptureEnabled = _isAutoCaptureEnabled.asStateFlow()

    private val _confidenceThreshold = MutableStateFlow(0.25f)
    val confidenceThreshold = _confidenceThreshold.asStateFlow()

    fun setAutoCapture(enabled: Boolean) {
        _isAutoCaptureEnabled.value = enabled
    }

    fun setConfidenceThreshold(value: Float) {
        _confidenceThreshold.value = value
    }

    fun setSaveToDevice(enabled: Boolean) {
        _saveToDevice.value = enabled
    }

    fun analyzeImage(imagePath: String) {
        viewModelScope.launch {
            _isAnalyzing.value = true
            pipeline.initialize()
            val result = pipeline.analyze(imagePath)
            _predictionResult.value = result
            _isAnalyzing.value = false

            if (saveToDevice.value && result.error == null) {
                val original = BitmapFactory.decodeFile(imagePath)
                if (original != null) {
                    ResultImageSaver.saveResultImage(getApplication(), original, result)
                    original.recycle()
                }
            }
        }
    }

    override fun onCleared() {
        super.onCleared()
        pipeline.close()
    }
}
