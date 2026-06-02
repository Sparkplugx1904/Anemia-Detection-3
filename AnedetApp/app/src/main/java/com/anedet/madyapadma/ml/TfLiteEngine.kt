package com.anedet.madyapadma.ml

import android.content.Context
import com.google.android.gms.tflite.client.TfLiteInitializationOptions
import com.google.android.gms.tflite.gpu.support.TfLiteGpu
import com.google.android.gms.tflite.java.TfLite
import kotlinx.coroutines.tasks.await
import org.tensorflow.lite.InterpreterApi
import org.tensorflow.lite.Delegate
import java.nio.MappedByteBuffer

/**
 * Encapsulates TFLite model loading and interpreter creation using Google Play Services.
 */
class TfLiteEngine(private val context: Context) {

    private var interpreter: InterpreterApi? = null
    private var gpuDelegate: Delegate? = null
    private var isInitialized = false

    suspend fun initialize() {
        if (isInitialized) return

        // 1. Initialize TFLite with Google Play Services
        TfLite.initialize(context, TfLiteInitializationOptions.builder()
            .setEnableGpuDelegateSupport(true)
            .build()).await()

        isInitialized = true
    }

    suspend fun createInterpreter(modelBuffer: MappedByteBuffer, useGpu: Boolean = true): InterpreterApi {
        if (!isInitialized) initialize()

        val options = InterpreterApi.Options()

        if (useGpu) {
            val gpuAvailable = TfLiteGpu.isGpuDelegateAvailable(context).await()
            if (gpuAvailable) {
                options.setRuntime(InterpreterApi.Options.TfLiteRuntime.FROM_SYSTEM_ONLY)
            } else {
                options.setNumThreads(4)
                options.setUseXNNPACK(true)
            }
        } else {
            options.setNumThreads(4)
            options.setUseXNNPACK(true)
        }

        val interp = InterpreterApi.create(modelBuffer, options)
        this.interpreter = interp
        return interp
    }

    fun close() {
        interpreter?.close()
        // gpuDelegate?.close() // Delegate might not have close if it's just an interface
        interpreter = null
        gpuDelegate = null
    }
}
