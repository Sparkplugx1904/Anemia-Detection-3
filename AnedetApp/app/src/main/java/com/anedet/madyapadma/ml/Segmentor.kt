package com.anedet.madyapadma.ml

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.RectF
import com.anedet.madyapadma.model.MaskData
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

class Segmentor(context: Context) {

    companion object {
        private const val MODEL_PATH = "yolo26n_seg_fp16.tflite"
        const val INPUT_SIZE = 320
        private const val CONF_THRESHOLD = 0.25f
    }

    private var interpreter: Interpreter? = null
    private var outputShapes = listOf<IntArray>()

    init {
        loadModel(context)
    }

    private fun loadModel(context: Context) {
        val modelBuffer = loadModelFile(context, MODEL_PATH)
        interpreter = TfLiteHelper.createInterpreter(modelBuffer)

        val numOutputs = interpreter?.outputTensorCount ?: 0
        outputShapes = (0 until numOutputs).map { idx ->
            interpreter?.getOutputTensor(idx)?.shape() ?: intArrayOf()
        }
    }

    fun runSegmentation(imagePath: String): MaskData? {
        val bitmap = BitmapFactory.decodeFile(imagePath) ?: return null
        val imgW = bitmap.width
        val imgH = bitmap.height

        val inputBuffer = preprocess(bitmap)
        bitmap.recycle()

        val outputs = runInference(inputBuffer) ?: return null
        return parseOutput(outputs, imgW, imgH)
    }

    private fun preprocess(bitmap: Bitmap): ByteBuffer {
        val resized = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, true)
        val pixels = IntArray(INPUT_SIZE * INPUT_SIZE)
        resized.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)
        resized.recycle()

        val inputChannels = 3
        val inputSize = 1 * INPUT_SIZE * INPUT_SIZE * inputChannels
        val buffer = ByteBuffer.allocateDirect(inputSize * 4)
        buffer.order(ByteOrder.nativeOrder())

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

    private fun runInference(inputBuffer: ByteBuffer): List<Array<Any>>? {
        val interp = interpreter ?: return null
        val numOutputs = outputShapes.size
        if (numOutputs == 0) return null

        val outputs = arrayOfNulls<Any>(numOutputs)
        for (i in 0 until numOutputs) {
            val shape = outputShapes[i]
            outputs[i] = when (shape.size) {
                3 -> Array(shape[0]) { Array(shape[1]) { FloatArray(shape[2]) } }
                4 -> Array(shape[0]) { Array(shape[1]) { Array(shape[2]) { FloatArray(shape[3]) } } }
                else -> FloatArray(shape.lastOrNull() ?: 1)
            }
        }

        val outputMap = HashMap<Int, Any>()
        for (i in 0 until numOutputs) {
            outputMap[i] = outputs[i] as Any
        }
        interp.runForMultipleInputsOutputs(arrayOf(inputBuffer), outputMap)

        return outputs.mapIndexed { i, out ->
            val sh = outputShapes[i]
            if (sh.size >= 3 && sh[0] == 1) {
                (out as Array<Any>)[0] as Array<Any>
            } else {
                out as Array<Any>
            }
        }
    }

    private fun parseOutput(outputs: List<Array<Any>>, imgW: Int, imgH: Int): MaskData? {
        if (outputs.isEmpty()) return null

        val detShape = outputShapes[0]
        val detRows = if (detShape.size >= 3) detShape[1] else 0
        val detCols = if (detShape.size >= 3) detShape[2] else 0

        if (detRows < 5) return null

        var bestConf = 0f
        var bestCol = -1

        for (col in 0 until detCols) {
            val conf = (outputs[0][4] as FloatArray)[col]
            if (conf > bestConf && conf > CONF_THRESHOLD) {
                bestConf = conf
                bestCol = col
            }
        }

        if (bestCol < 0) return null

        val x1 = (outputs[0][0] as FloatArray)[bestCol] / INPUT_SIZE.toFloat() * imgW
        val y1 = (outputs[0][1] as FloatArray)[bestCol] / INPUT_SIZE.toFloat() * imgH
        val x2 = (outputs[0][2] as FloatArray)[bestCol] / INPUT_SIZE.toFloat() * imgW
        val y2 = (outputs[0][3] as FloatArray)[bestCol] / INPUT_SIZE.toFloat() * imgH

        val bbox = RectF(x1, y1, x2, y2)

        if (outputs.size < 2) {
            return MaskData(bbox, fallbackMask(bbox, imgW, imgH), bestConf)
        }

        val maskShape = outputShapes[1]
        val protoH = if (maskShape.size >= 3) maskShape[1] else 1
        val protoW = if (maskShape.size >= 3) maskShape[2] else 1

        val numCoeffs = detRows - 5
        val maskCoeffs = FloatArray(numCoeffs.coerceAtLeast(1))
        for (c in 0 until maskCoeffs.size) {
            if (5 + c < detRows) {
                maskCoeffs[c] = (outputs[0][5 + c] as FloatArray)[bestCol]
            }
        }

        val maskOut = Array(imgH) { FloatArray(imgW) }
        val protoChannels = outputShapes[1].getOrElse(0) { 1 }

        for (y in 0 until imgH) {
            for (x in 0 until imgW) {
                val py = (y * protoH) / imgH
                val px = (x * protoW) / imgW
                var sum = 0f
                for (c in 0 until maskCoeffs.size.coerceAtMost(protoChannels)) {
                    val protoVal = when (val proto = outputs[1][c]) {
                        is FloatArray -> proto[py * protoW + px]
                        else -> 0f
                    }
                    sum += maskCoeffs[c] * protoVal
                }
                val sigmoid = 1.0f / (1.0f + kotlin.math.exp(-sum))
                maskOut[y][x] = if (sigmoid > 0.5f) 1f else 0f
            }
        }

        return MaskData(bbox, maskOut, bestConf)
    }

    private fun fallbackMask(bbox: RectF, w: Int, h: Int): Array<FloatArray> {
        val mask = Array(h) { FloatArray(w) }
        val cx = bbox.centerX().toInt()
        val cy = bbox.centerY().toInt()
        val rx = (bbox.width() / 2f).toInt().coerceAtLeast(1)
        val ry = (bbox.height() / 2f).toInt().coerceAtLeast(1)
        for (y in 0 until h) {
            for (x in 0 until w) {
                val dx = x - cx
                val dy = y - cy
                mask[y][x] = if (dx * dx * ry * ry + dy * dy * rx * rx <= rx * rx * ry * ry) 1f else 0f
            }
        }
        return mask
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
