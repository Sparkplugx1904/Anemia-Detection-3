package com.anedet.madyapadma.ml

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.RectF
import android.util.Log
import com.anedet.madyapadma.model.MaskData
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

/**
 * Segmentor berbasis YOLO-seg (YOLOv8 / YOLO26) dengan GPU acceleration.
 * Disesuaikan dengan ground-truth model: 38 features, 80x80 proto.
 */
class Segmentor(private val context: Context) {

    companion object {
        private const val MODEL_PATH = "yolo26n_seg_fp16.tflite"
        const val INPUT_SIZE = 320
        private const val CONF_THRESHOLD = 0.25f
        private const val LETTERBOX_FILL = 0.5f
        private const val TAG = "Segmentor"
    }

    private var interpreter: Interpreter? = null
    private var gpuDelegate: GpuDelegate? = null
    private var outputShapes = listOf<IntArray>()
    private var outputBuffers = mutableMapOf<Int, ByteBuffer>()

    private val lock = Any()

    private val inputBuffer: ByteBuffer = ByteBuffer
        .allocateDirect(1 * INPUT_SIZE * INPUT_SIZE * 3 * 4)
        .also { it.order(ByteOrder.nativeOrder()) }

    private data class LetterboxParams(
        val scale: Float,
        val padLeft: Int,
        val padTop: Int,
        val origW: Int,
        val origH: Int
    )

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

        val interp = Interpreter(modelBuffer, options)
        interpreter = interp

        val numOutputs = interp.outputTensorCount
        val shapes = mutableListOf<IntArray>()
        val buffers = mutableMapOf<Int, ByteBuffer>()

        for (i in 0 until numOutputs) {
            val tensor = interp.getOutputTensor(i)
            val shape = tensor.shape()
            shapes.add(shape)
            val buffer = ByteBuffer.allocateDirect(tensor.numBytes()).apply { order(ByteOrder.nativeOrder()) }
            buffers[i] = buffer
        }
        
        outputShapes = shapes
        outputBuffers = buffers
        Log.i(TAG, "Segmentor initialized with shapes=${shapes.map { it.toList() }}")
    }

    fun isReady(): Boolean = interpreter != null && outputBuffers.isNotEmpty()

    fun runSegmentation(bitmap: Bitmap): MaskData? = synchronized(lock) {
        val interp = interpreter ?: return null
        if (outputBuffers.isEmpty()) return null
        
        val lbParams = preprocess(bitmap)
        runInference(interp)
        return parseOutput(lbParams, bitmap.width, bitmap.height)
    }

    private fun preprocess(bitmap: Bitmap): LetterboxParams {
        val origW = bitmap.width; val origH = bitmap.height
        val scale = min(INPUT_SIZE.toFloat() / origW, INPUT_SIZE.toFloat() / origH)
        val scaledW = (origW * scale).toInt(); val scaledH = (origH * scale).toInt()
        val padLeft = (INPUT_SIZE - scaledW) / 2; val padTop  = (INPUT_SIZE - scaledH) / 2

        val scaledBitmap = Bitmap.createScaledBitmap(bitmap, scaledW, scaledH, true)
        val pixels = IntArray(scaledW * scaledH)
        scaledBitmap.getPixels(pixels, 0, scaledW, 0, 0, scaledW, scaledH)
        scaledBitmap.recycle()

        inputBuffer.rewind()
        val fillVal = LETTERBOX_FILL
        for (y in 0 until INPUT_SIZE) {
            for (x in 0 until INPUT_SIZE) {
                val srcX = x - padLeft; val srcY = y - padTop
                if (srcX < 0 || srcX >= scaledW || srcY < 0 || srcY >= scaledH) {
                    inputBuffer.putFloat(fillVal); inputBuffer.putFloat(fillVal); inputBuffer.putFloat(fillVal)
                } else {
                    val pixel = pixels[srcY * scaledW + srcX]
                    inputBuffer.putFloat(((pixel shr 16) and 0xFF) / 255.0f)
                    inputBuffer.putFloat(((pixel shr 8)  and 0xFF) / 255.0f)
                    inputBuffer.putFloat(( pixel         and 0xFF) / 255.0f)
                }
            }
        }
        inputBuffer.rewind()
        return LetterboxParams(scale, padLeft, padTop, origW, origH)
    }

    private fun runInference(interp: Interpreter) {
        val outputMap = HashMap<Int, Any>()
        for (i in outputShapes.indices) {
            val buf = outputBuffers[i] ?: continue
            buf.rewind(); outputMap[i] = buf
        }
        interp.runForMultipleInputsOutputs(arrayOf(inputBuffer), outputMap)
    }

    private fun parseOutput(lbParams: LetterboxParams, origW: Int, origH: Int): MaskData? {
        val detRawBuf = outputBuffers[0] ?: return null
        detRawBuf.rewind()
        val detBuf = detRawBuf.asFloatBuffer()
        
        val detShape = outputShapes[0] // [1, 300, 38]
        if (detShape.size < 3) return null
        
        val numAnchors = detShape[1]
        val numFeatures = detShape[2]

        if (numFeatures < 38 || numAnchors == 0) return null

        var bestConf = 0f; var bestIdx = -1
        for (ancIdx in 0 until numAnchors) {
            // Ground Truth: Index 4 is confidence/score
            val conf = detBuf.get(ancIdx * numFeatures + 4)
            if (conf > bestConf && conf > CONF_THRESHOLD) {
                bestConf = conf; bestIdx = ancIdx
            }
        }
        if (bestIdx < 0) return null

        // Parse x1,y1,x2,y2 directly (Ground Truth: nms_coord_format=x1y1x2y2)
        var x1Raw = detBuf.get(bestIdx * numFeatures + 0)
        var y1Raw = detBuf.get(bestIdx * numFeatures + 1)
        var x2Raw = detBuf.get(bestIdx * numFeatures + 2)
        var y2Raw = detBuf.get(bestIdx * numFeatures + 3)
        
        if (x1Raw <= 1.01f && x2Raw <= 1.01f) {
            x1Raw *= INPUT_SIZE; x2Raw *= INPUT_SIZE
            y1Raw *= INPUT_SIZE; y2Raw *= INPUT_SIZE
        }

        val bbox = decodeBbox(x1Raw, y1Raw, x2Raw, y2Raw, lbParams)

        // Ground Truth: Coefficients start at index 6 (4=conf, 5=class0)
        val coeffs = FloatArray(32)
        for (k in 0 until 32) {
            coeffs[k] = detBuf.get(bestIdx * numFeatures + (6 + k))
        }

        val protoRawBuf = outputBuffers[1] ?: return null
        protoRawBuf.rewind()
        val protoBuf = protoRawBuf.asFloatBuffer()
        val protoShape = outputShapes[1] // [1, 80, 80, 32]
        
        val mask = computeProtoMask(protoBuf, protoShape, coeffs, bbox, lbParams)

        // Detect correct resolution from NHWC [1, 80, 80, 32] or NCHW [1, 32, 80, 80]
        val mw = if (protoShape[1] != 32) protoShape[2] else protoShape[3]
        val mh = if (protoShape[1] != 32) protoShape[1] else protoShape[2]

        return MaskData(
            bbox = bbox,
            mask = mask,
            confidence = bestConf,
            protoW = mw,
            protoH = mh,
            lbScale = lbParams.scale,
            lbPadLeft = lbParams.padLeft,
            lbPadTop = lbParams.padTop
        )
    }

    private fun computeProtoMask(
        protoBuf: java.nio.FloatBuffer,
        protoShape: IntArray,
        coeffs: FloatArray,
        bbox: RectF,
        lb: LetterboxParams
    ): Array<FloatArray> {
        // NHWC [1, 80, 80, 32] or NCHW [1, 32, 80, 80]
        val isNHWC = protoShape[1] != 32
        val h = if (isNHWC) protoShape[1] else protoShape[2]
        val w = if (isNHWC) protoShape[2] else protoShape[3]
        val c = if (isNHWC) protoShape[3] else protoShape[1]

        // Map bbox to proto space (80x80)
        val pLeft   = ((bbox.left   * lb.scale + lb.padLeft) * (w / INPUT_SIZE.toFloat())).toInt().coerceIn(0, w - 1)
        val pTop    = ((bbox.top    * lb.scale + lb.padTop)  * (h / INPUT_SIZE.toFloat())).toInt().coerceIn(0, h - 1)
        val pRight  = ((bbox.right  * lb.scale + lb.padLeft) * (w / INPUT_SIZE.toFloat())).toInt().coerceIn(0, w - 1)
        val pBottom = ((bbox.bottom * lb.scale + lb.padTop)  * (h / INPUT_SIZE.toFloat())).toInt().coerceIn(0, h - 1)

        val result = Array(h) { FloatArray(w) }
        for (y in 0 until h) {
            for (x in 0 until w) {
                if (x < pLeft || x > pRight || y < pTop || y > pBottom) {
                    result[y][x] = 0f
                    continue
                }

                var sum = 0f
                if (isNHWC) {
                    val base = (y * w + x) * c
                    for (k in 0 until 32) sum += protoBuf.get(base + k) * coeffs[k]
                } else {
                    val plane = h * w
                    for (k in 0 until 32) sum += protoBuf.get(k * plane + y * w + x) * coeffs[k]
                }
                result[y][x] = if (sigmoid(sum) > 0.5f) 1f else 0f
            }
        }
        return result
    }

    private fun sigmoid(x: Float): Float = (1f / (1f + exp(-x.toDouble()))).toFloat()

    private fun decodeBbox(x1m: Float, y1m: Float, x2m: Float, y2m: Float, p: LetterboxParams): RectF {
        val x1 = ((x1m - p.padLeft) / p.scale).coerceIn(0f, p.origW.toFloat())
        val y1 = ((y1m - p.padTop)  / p.scale).coerceIn(0f, p.origH.toFloat())
        val x2 = ((x2m - p.padLeft) / p.scale).coerceIn(0f, p.origW.toFloat())
        val y2 = ((y2m - p.padTop)  / p.scale).coerceIn(0f, p.origH.toFloat())
        return RectF(x1, y1, x2, y2)
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

    suspend fun runSegmentation(imagePath: String): MaskData? {
        if (interpreter == null) initialize()
        val bitmap = BitmapFactory.decodeFile(imagePath) ?: return null
        return runSegmentation(bitmap)
    }
}