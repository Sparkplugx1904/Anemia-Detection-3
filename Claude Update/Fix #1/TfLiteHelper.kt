package com.anedet.madyapadma.ml

import android.util.Log
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import java.nio.MappedByteBuffer

/**
 * Helper untuk membuat TFLite Interpreter dengan GPU delegate (jika didukung)
 * atau fallback ke CPU multi-thread.
 *
 * Strategi:
 *   1. Cek CompatibilityList — apakah device mendukung GPU delegate
 *   2. Jika ya: pakai GpuDelegate dengan bestOptionsForThisDevice + setPrecisionLossAllowed(true)
 *      (FP16 quantization di GPU untuk model FP16 lebih cepat)
 *   3. Jika tidak: fallback ke 4 CPU threads
 *
 * Caller bertanggung jawab menutup GpuDelegate bersamaan dengan Interpreter.close().
 */
object TfLiteHelper {

    private const val TAG = "TfLiteHelper"
    private const val CPU_THREADS = 4

    data class InterpreterBundle(
        val interpreter: Interpreter,
        val gpuDelegate: GpuDelegate? = null
    ) {
        fun close() {
            interpreter.close()
            gpuDelegate?.close()
        }
    }

    /**
     * Buat Interpreter dengan GPU delegate jika tersedia.
     * Kembalikan [InterpreterBundle] — pastikan dipanggil [close()] saat tidak dipakai.
     */
    fun createInterpreter(modelBuffer: MappedByteBuffer): InterpreterBundle {
        val compatList = CompatibilityList()

        return if (compatList.isDelegateSupportedOnThisDevice) {
            try {
                val gpuOptions = compatList.bestOptionsForThisDevice.apply {
                    isPrecisionLossAllowed = true  // Izinkan FP16 → lebih cepat di GPU
                }
                val delegate = GpuDelegate(gpuOptions)
                val options = Interpreter.Options().apply {
                    addDelegate(delegate)
                    // Tetap set threads sebagai fallback jika ada op yang tidak didukung GPU
                    numThreads = 2
                }
                Log.i(TAG, "GPU delegate aktif")
                InterpreterBundle(Interpreter(modelBuffer, options), delegate)
            } catch (e: Exception) {
                Log.w(TAG, "GPU delegate gagal diinisialisasi, fallback ke CPU: ${e.message}")
                createCpuInterpreter(modelBuffer)
            }
        } else {
            Log.i(TAG, "GPU delegate tidak didukung, pakai CPU $CPU_THREADS threads")
            createCpuInterpreter(modelBuffer)
        }
    }

    private fun createCpuInterpreter(modelBuffer: MappedByteBuffer): InterpreterBundle {
        val options = Interpreter.Options().apply {
            numThreads = CPU_THREADS
            // XNNPACK delegate untuk akselerasi CPU (aktif secara default di TFLite 2.14+,
            // tapi eksplisit lebih aman)
            setUseXNNPACK(true)
        }
        return InterpreterBundle(Interpreter(modelBuffer, options))
    }
}
