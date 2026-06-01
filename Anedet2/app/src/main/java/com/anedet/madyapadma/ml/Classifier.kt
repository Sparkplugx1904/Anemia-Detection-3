package com.anedet.madyapadma.ml

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

class Classifier(context: Context) {

    companion object {
        private const val MODEL_PATH = "yolo26s_cls_fp16.tflite"
        const val INPUT_SIZE = 448
    }

    private var interpreter: Interpreter? = null

    init {
        loadModel(context)
    }

    private fun loadModel(context: Context) {
        val modelBuffer = loadModelFile(context, MODEL_PATH)
        interpreter = TfLiteHelper.createInterpreter(modelBuffer)
    }

    fun classify(imagePath: String): Pair<Float, Float>? {
        val bitmap = BitmapFactory.decodeFile(imagePath) ?: return null
        val resized = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, true)

        val inputBuffer = preprocess(resized)
        resized.recycle()
        bitmap.recycle()

        val outputShape = interpreter?.getOutputTensor(0)?.shape() ?: intArrayOf(1, 2)
        val numClasses = outputShape.getOrElse(1) { 2 }

        val output = Array(1) { FloatArray(numClasses) }
        interpreter?.run(inputBuffer, output)

        val logits = output[0]
        val maxLogit = logits.max()
        val expSum = logits.sumOf { kotlin.math.exp((it - maxLogit).toDouble()) }.toFloat()
        val probs = logits.map { kotlin.math.exp(it - maxLogit) / expSum }

        return Pair(probs.getOrElse(0) { 0f }, probs.getOrElse(1) { 0f })
    }

    private fun preprocess(bitmap: Bitmap): ByteBuffer {
        val inputSize = 1 * INPUT_SIZE * INPUT_SIZE * 3
        val buffer = ByteBuffer.allocateDirect(inputSize * 4)
        buffer.order(ByteOrder.nativeOrder())

        val pixels = IntArray(INPUT_SIZE * INPUT_SIZE)
        bitmap.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)
        for (pixel in pixels) {
            val r = ((pixel shr 16) and 0xFF) / 255.0f
            val g = ((pixel shr 8) and 0xFF) / 255.0f
            val b = (pixel and 0xFF) / 255.0f
            buffer.putFloat(r)
            buffer.putFloat(g)
            buffer.putFloat(b)
        }
        buffer.rewind()
        return buffer
    }

    private fun loadModelFile(context: Context, modelPath: String): MappedByteBuffer {
        val assetFd = context.assets.openFd(modelPath)
        val inputStream = FileInputStream(assetFd.fileDescriptor)
        val fileChannel = inputStream.channel
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, assetFd.startOffset, assetFd.declaredLength)
    }

    fun close() {
        interpreter?.close()
        interpreter = null
    }
}
