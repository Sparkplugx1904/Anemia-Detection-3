package com.anedet.madyapadma.camera

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.anedet.madyapadma.ml.AnemiaPipeline
import com.anedet.madyapadma.model.PredictionResult
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch

class CameraViewModel(application: Application) : AndroidViewModel(application) {

    private val pipeline = AnemiaPipeline(application)

    private val _predictionResult = MutableStateFlow<PredictionResult?>(null)
    val predictionResult: StateFlow<PredictionResult?> = _predictionResult

    private val _isAnalyzing = MutableStateFlow(false)
    val isAnalyzing: StateFlow<Boolean> = _isAnalyzing

    fun analyzeImage(imagePath: String) {
        viewModelScope.launch {
            _isAnalyzing.value = true
            _predictionResult.value = null
            try {
                val result = pipeline.analyze(imagePath)
                _predictionResult.value = result
            } catch (e: Exception) {
                _predictionResult.value = PredictionResult(
                    isAnemic = false,
                    anemicProbability = 0f,
                    nonAnemicProbability = 0f,
                    maskOverlay = null,
                    inferenceTimeMs = 0,
                    error = e.message ?: "Unknown error"
                )
            } finally {
                _isAnalyzing.value = false
            }
        }
    }

    override fun onCleared() {
        super.onCleared()
        pipeline.close()
    }
}
