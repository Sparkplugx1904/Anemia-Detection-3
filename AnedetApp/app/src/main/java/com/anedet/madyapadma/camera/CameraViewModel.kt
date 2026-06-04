package com.anedet.madyapadma.camera

import android.app.Application
import android.graphics.BitmapFactory
import android.util.Log
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.anedet.madyapadma.ml.AnemiaPipeline
import com.anedet.madyapadma.ml.ResultImageSaver
import com.anedet.madyapadma.model.MaskData
import com.anedet.madyapadma.model.PredictionResult
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asSharedFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

class CameraViewModel(application: Application) : AndroidViewModel(application) {

    private val pipeline = AnemiaPipeline(application)
    val settings = AppSettings(application)

    private val _predictionResult = MutableStateFlow<PredictionResult?>(null)
    val predictionResult: StateFlow<PredictionResult?> = _predictionResult.asStateFlow()

    private val _isAnalyzing = MutableStateFlow(false)
    val isAnalyzing: StateFlow<Boolean> = _isAnalyzing.asStateFlow()

    private val _lastLiveMask = MutableStateFlow<MaskData?>(null)
    val lastLiveMask: StateFlow<MaskData?> = _lastLiveMask.asStateFlow()

    private val _liveImageW = MutableStateFlow(0)
    val liveImageW: StateFlow<Int> = _liveImageW.asStateFlow()

    private val _liveImageH = MutableStateFlow(0)
    val liveImageH: StateFlow<Int> = _liveImageH.asStateFlow()

    /**
     * Auto-capture diminta dari capture screen.
     * CaptureScreen subscribe ke event ini dan memanggil takePicture().
     */
    private val _autoCaptureRequests = MutableSharedFlow<String>(extraBufferCapacity = 1)
    val autoCaptureRequests: SharedFlow<String> = _autoCaptureRequests.asSharedFlow()

    /**
     * Status smart auto-capture (untuk UI):
     *  - "searching"  : mencari konjungtiva
     *  - "stabilizing": terdeteksi, menunggu frame stabil
     *  - "capturing"  : sedang ambil foto
     *  - "low_quality": terdeteksi tapi blur / area kecil
     *  - "ready"      : sudah stabil, trigger capture
     */
    private val _autoCaptureStatus = MutableStateFlow("searching")
    val autoCaptureStatus: StateFlow<String> = _autoCaptureStatus.asStateFlow()

    private val _autoCaptureProgress = MutableStateFlow(0)
    val autoCaptureProgress: StateFlow<Int> = _autoCaptureProgress.asStateFlow()

    fun updateLiveMask(data: MaskData?, imgW: Int, imgH: Int) {
        _lastLiveMask.value = data
        _liveImageW.value = imgW
        _liveImageH.value = imgH
    }

    fun reportAutoCaptureStatus(status: String, progress: Int = 0) {
        _autoCaptureStatus.value = status
        _autoCaptureProgress.value = progress
    }

    /**
     * Dipanggil dari CaptureScreen saat smart auto-capture sudah siap trigger.
     * Mengirim event agar CaptureScreen menjalankan takePicture.
     */
    fun requestAutoCapture(captureDir: String) {
        viewModelScope.launch {
            _autoCaptureRequests.emit(captureDir)
        }
    }

    fun analyzeImage(imagePath: String) {
        viewModelScope.launch {
            _isAnalyzing.value = true
            pipeline.initialize()
            val result = pipeline.analyze(imagePath)
            _predictionResult.value = result
            _isAnalyzing.value = false
        }
    }

    fun saveResultToGallery(imagePath: String) {
        val result = _predictionResult.value ?: return
        if (result.error != null) {
            Log.w(TAG, "saveResultToGallery: result has error, skip")
            return
        }
        viewModelScope.launch {
            val original = BitmapFactory.decodeFile(imagePath)
            if (original != null) {
                val ok = ResultImageSaver.saveResultImage(getApplication(), original, result)
                Log.d(TAG, "saveResultToGallery: saved=$ok")
                original.recycle()
            }
        }
    }

    override fun onCleared() {
        super.onCleared()
        pipeline.close()
    }

    companion object {
        private const val TAG = "CameraViewModel"
    }
}
