package com.anedet.madyapadma.ml

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import android.util.Log
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.exp
import kotlin.math.min
import kotlin.math.pow

/**
 * Classifier konjungtiva anemia.
 * Output: class 0 = Anemia, class 1 = Non-Anemia
 */
class Classifier(private val context: Context) {

    companion object {
        private const val TAG = "Classifier"
        private const val MODEL_PATH = "yolo26s_cls_fp16.tflite"
        const val INPUT_SIZE = 448
    }

    private var interpreter: Interpreter? = null
    private var gpuDelegate: GpuDelegate? = null

    // Pre-allocated buffer
    private val inputBuffer: ByteBuffer = ByteBuffer
        .allocateDirect(1 * INPUT_SIZE * INPUT_SIZE * 3 * 4)
        .also { it.order(ByteOrder.nativeOrder()) }

    suspend fun initialize() {
        if (interpreter != null) return
        val modelBuffer = loadModelFile(this.context, MODEL_PATH)
        val options = Interpreter.Options().apply {
            val compatList = CompatibilityList()
            if (compatList.isDelegateSupportedOnThisDevice) {
                gpuDelegate = GpuDelegate(compatList.bestOptionsForThisDevice)
                addDelegate(gpuDelegate)
            } else {
                setNumThreads(4)
                setUseXNNPACK(true)
            }
        }

        interpreter = Interpreter(modelBuffer, options)
        Log.i(TAG, "Classifier initialized: output shape=${interpreter?.getOutputTensor(0)?.shape()?.toList()}")
    }

    /**
     * Klasifikasi dari Bitmap yang sudah di-crop dari region konjungtiva.
     */
    suspend fun classify(bitmap: Bitmap): Pair<Float, Float>? {
        if (interpreter == null) initialize()
        return try {
            preprocessToBuffer(bitmap)
            runClassification()
        } catch (e: Exception) {
            null
        }
    }

    private fun preprocessToBuffer(src: Bitmap) {
        val (lbBitmap, _, _) = letterbox(src, INPUT_SIZE)
        inputBuffer.rewind()
        val pixels = IntArray(INPUT_SIZE * INPUT_SIZE)
        lbBitmap.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)
        lbBitmap.recycle()

        for (pixel in pixels) {
            inputBuffer.putFloat(((pixel shr 16) and 0xFF) / 255.0f)
            inputBuffer.putFloat(((pixel shr 8)  and 0xFF) / 255.0f)
            inputBuffer.putFloat(( pixel         and 0xFF) / 255.0f)
        }
        inputBuffer.rewind()
    }

    private fun runClassification(): Pair<Float, Float>? {
        val interp = interpreter ?: return null
        val outputShape = interp.getOutputTensor(0).shape()
        val numClasses = outputShape.getOrElse(1) { 2 }
        val output = Array(1) { FloatArray(numClasses) }

        val outputMap = HashMap<Int, Any>()
        outputMap[0] = output
        interp.runForMultipleInputsOutputs(arrayOf(inputBuffer), outputMap)

        val logits = output[0]
        val sum = logits.sum()
        if (sum > 0.95f && sum < 1.05f) {
            return Pair(logits.getOrElse(0) { 0f }, logits.getOrElse(1) { 0f })
        }

        val maxLogit = logits.max()
        val expVals  = logits.map { exp((it - maxLogit).toDouble()) }
        val expSum   = expVals.sum()
        val probs    = expVals.map { (it / expSum).toFloat() }

        return Pair(probs.getOrElse(0) { 0f }, probs.getOrElse(1) { 0f })
    }

    private fun letterbox(src: Bitmap, targetSize: Int): Triple<Bitmap, Int, Int> {
        val origW = src.width; val origH = src.height
        val scale  = min(targetSize.toFloat() / origW, targetSize.toFloat() / origH)
        val scaledW = (origW * scale).toInt()
        val scaledH = (origH * scale).toInt()
        val padLeft = (targetSize - scaledW) / 2
        val padTop  = (targetSize - scaledH) / 2

        val scaled = Bitmap.createScaledBitmap(src, scaledW, scaledH, true)
        val result = Bitmap.createBitmap(targetSize, targetSize, Bitmap.Config.ARGB_8888)
        val canvas = android.graphics.Canvas(result)
        canvas.drawColor(android.graphics.Color.rgb(128, 128, 128))
        canvas.drawBitmap(scaled, padLeft.toFloat(), padTop.toFloat(), null)
        scaled.recycle()
        return Triple(result, padLeft, padTop)
    }

    private fun loadModelFile(context: Context, modelPath: String): MappedByteBuffer {
        val assetFd = context.assets.openFd(modelPath)
        return FileInputStream(assetFd.fileDescriptor).channel.map(
            FileChannel.MapMode.READ_ONLY, assetFd.startOffset, assetFd.declaredLength
        )
    }

    fun close() {
        interpreter?.close(); interpreter = null
        gpuDelegate?.close(); gpuDelegate = null
    }
}
