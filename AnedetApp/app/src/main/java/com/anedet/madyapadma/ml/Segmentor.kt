package com.anedet.madyapadma.ml

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.RectF
import com.anedet.madyapadma.model.MaskData
import org.tensorflow.lite.InterpreterApi
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.exp
import kotlin.math.min

class Segmentor(private val context: Context) {

    companion object {
        private const val MODEL_PATH = "yolo26n_seg_fp16.tflite"
        const val INPUT_SIZE = 320
        private const val CONF_THRESHOLD = 0.25f
        // Nilai fill letterbox (0.5 = gray, sesuai YOLO training default)
        private const val LETTERBOX_FILL = 0.5f
    }

    private var engine: TfLiteEngine = TfLiteEngine(context)
    private var interpreter: InterpreterApi? = null
    private var outputShapes = listOf<IntArray>()

    private val lock = Any()

    // Pre-allocated input buffer — hindari alokasi heap setiap inference
    private val inputBuffer: ByteBuffer = ByteBuffer
        .allocateDirect(1 * INPUT_SIZE * INPUT_SIZE * 3 * 4)
        .also { it.order(ByteOrder.nativeOrder()) }

    // Menyimpan parameter letterbox untuk decode bbox
    private data class LetterboxParams(
        val scale: Float,
        val padLeft: Int,
        val padTop: Int,
        val origW: Int,
        val origH: Int
    )

    suspend fun initialize() {
        if (interpreter != null) return
        engine.initialize()
        val modelBuffer = loadModelFile(this.context, MODEL_PATH)
        interpreter = engine.createInterpreter(modelBuffer, useGpu = true)

        val interp = interpreter ?: throw RuntimeException("Failed to create interpreter")
        val numOutputs = interp.outputTensorCount
        outputShapes = (0 until numOutputs).map { idx ->
            interp.getOutputTensor(idx)?.shape() ?: intArrayOf()
        }
    }

    suspend fun runSegmentation(imagePath: String): MaskData? {
        if (interpreter == null) initialize()
        val bitmap = BitmapFactory.decodeFile(imagePath) ?: return null
        return runSegmentation(bitmap)
    }

    fun runSegmentation(bitmap: Bitmap): MaskData? = synchronized(lock) {
        val interp = interpreter ?: return null
        val lbParams = preprocess(bitmap)

        val outputs = runInference() ?: return null
        return parseOutput(outputs, lbParams)
    }

    /**
     * Letterbox preprocessing: skala seragam, pad sisi pendek dengan LETTERBOX_FILL.
     * Menyimpan parameter ke LetterboxParams untuk decode koordinat bbox.
     */
    private fun preprocess(bitmap: Bitmap): LetterboxParams {
        val origW = bitmap.width
        val origH = bitmap.height

        // Hitung scale seragam
        val scale = min(INPUT_SIZE.toFloat() / origW, INPUT_SIZE.toFloat() / origH)
        val scaledW = (origW * scale).toInt()
        val scaledH = (origH * scale).toInt()

        // Padding simetris
        val padLeft = (INPUT_SIZE - scaledW) / 2
        val padTop  = (INPUT_SIZE - scaledH) / 2

        // Resize bitmap ke ukuran scaled
        val scaledBitmap = Bitmap.createScaledBitmap(bitmap, scaledW, scaledH, true)

        inputBuffer.rewind()
        val fillVal = LETTERBOX_FILL

        for (y in 0 until INPUT_SIZE) {
            for (x in 0 until INPUT_SIZE) {
                val srcX = x - padLeft
                val srcY = y - padTop

                if (srcX < 0 || srcX >= scaledW || srcY < 0 || srcY >= scaledH) {
                    // Daerah padding
                    inputBuffer.putFloat(fillVal)
                    inputBuffer.putFloat(fillVal)
                    inputBuffer.putFloat(fillVal)
                } else {
                    val pixel = scaledBitmap.getPixel(srcX, srcY)
                    inputBuffer.putFloat(((pixel shr 16) and 0xFF) / 255.0f)
                    inputBuffer.putFloat(((pixel shr 8)  and 0xFF) / 255.0f)
                    inputBuffer.putFloat(( pixel         and 0xFF) / 255.0f)
                }
            }
        }
        inputBuffer.rewind()

        scaledBitmap.recycle()
        return LetterboxParams(scale, padLeft, padTop, origW, origH)
    }

    private fun runInference(): List<Array<Any>>? {
        val interp = interpreter ?: return null
        val numOutputs = outputShapes.size
        if (numOutputs == 0) return null

        val outputs = arrayOfNulls<Any>(numOutputs)
        for (i in 0 until numOutputs) {
            val shape = outputShapes[i]
            // We strip batch [1] here directly in allocation if possible or keep it
            outputs[i] = when (shape.size) {
                2 -> Array(shape[0]) { FloatArray(shape[1]) }
                3 -> Array(shape[0]) { Array(shape[1]) { FloatArray(shape[2]) } }
                4 -> Array(shape[0]) { Array(shape[1]) { Array(shape[2]) { FloatArray(shape[3]) } } }
                else -> FloatArray(shape.lastOrNull() ?: 1)
            }
        }

        val outputMap = HashMap<Int, Any>()
        for (i in 0 until numOutputs) outputMap[i] = outputs[i] as Any

        interp.runForMultipleInputsOutputs(arrayOf(inputBuffer), outputMap)

        // Strip batch dimension (index 0) dari setiap output
        return outputs.mapIndexed { i, out ->
            val sh = outputShapes[i]
            if (sh.isNotEmpty() && sh[0] == 1) {
                if (out is Array<*>) {
                    out[0] as Array<Any>
                } else if (out is FloatArray) {
                    // This case shouldn't happen with our logic but for safety
                    out
                } else {
                    out as Any
                }
            } else {
                out as Any
            }
        }.filterIsInstance<Array<Any>>()
    }

    /**
     * Parse output YOLO-seg:
     *   outputs[0]: detection tensor [1, 300, 38] -> stripped batch [300, 38]
     *
     * Format YOLOv8-seg [300, 38]:
     *  0-3: bbox (cx, cy, w, h) - Perlu konversi ke x1y1x2y2
     *  4: confidence
     *  5: class (eyelid/conjunctiva, etc - model ini khusus conjunctiva)
     *  6-37: mask coefficients (32)
     */
    private fun parseOutput(outputs: List<Array<Any>>, lbParams: LetterboxParams): MaskData? {
        if (outputs.isEmpty()) return null

        val det0 = outputs[0] as Array<FloatArray> // [300][38]
        val numAnchors = det0.size
        val numFeatures = det0[0].size

        if (numFeatures < 38 || numAnchors == 0) return null

        // Cari deteksi dengan confidence tertinggi (index 4)
        var bestConf = 0f
        var bestIdx  = -1

        for (ancIdx in 0 until numAnchors) {
            val conf = det0[ancIdx][4]
            if (conf > bestConf && conf > CONF_THRESHOLD) {
                bestConf = conf
                bestIdx  = ancIdx
            }
        }

        if (bestIdx < 0) return null

        // Ambil koordinat cx, cy, w, h (skala 320)
        val cx = det0[bestIdx][0]
        val cy = det0[bestIdx][1]
        val w  = det0[bestIdx][2]
        val h  = det0[bestIdx][3]

        val x1Model = cx - w/2f
        val y1Model = cy - h/2f
        val x2Model = cx + w/2f
        val y2Model = cy + h/2f

        // Decode dari skala model ke koordinat gambar original (kompensasi letterbox)
        val bbox = decodeBbox(x1Model, y1Model, x2Model, y2Model, lbParams)

        // Model ini hanya punya 1 output [1, 300, 38] sesuai keluhan poin 3 & 11.
        // Tidak ada prototype mask terpisah.
        // Menggunakan ellipse fallback atau mask di-reconstruct dari coeff jika model support (biasanya YOLO-seg butuh proto)
        // Sesuai Keluhan 11: "output tensor hanya 1 buah ([1,300,38]) tanpa prototype mask terpisah"

        return MaskData(bbox, fallbackMask(bbox, lbParams.origW, lbParams.origH), bestConf)
    }

    /**
     * Decode koordinat dari skala model (INPUT_SIZE) ke koordinat gambar original,
     * dengan kompensasi letterbox padding.
     */
    private fun decodeBbox(
        x1m: Float, y1m: Float, x2m: Float, y2m: Float,
        p: LetterboxParams
    ): RectF {
        // Hapus padding, kemudian bagi scale
        val x1 = ((x1m - p.padLeft) / p.scale).coerceIn(0f, p.origW.toFloat())
        val y1 = ((y1m - p.padTop)  / p.scale).coerceIn(0f, p.origH.toFloat())
        val x2 = ((x2m - p.padLeft) / p.scale).coerceIn(0f, p.origW.toFloat())
        val y2 = ((y2m - p.padTop)  / p.scale).coerceIn(0f, p.origH.toFloat())
        return RectF(x1, y1, x2, y2)
    }

    private fun fallbackMask(bbox: RectF, w: Int, h: Int): Array<FloatArray> {
        val mask = Array(h) { FloatArray(w) }
        val cx = bbox.centerX().toInt()
        val cy = bbox.centerY().toInt()
        val rx = (bbox.width()  / 2f).toInt().coerceAtLeast(1)
        val ry = (bbox.height() / 2f).toInt().coerceAtLeast(1)
        val rxSq = rx.toLong() * rx
        val rySq = ry.toLong() * ry
        for (y in 0 until h) {
            for (x in 0 until w) {
                val dx = (x - cx).toLong()
                val dy = (y - cy).toLong()
                mask[y][x] = if (dx * dx * rySq + dy * dy * rxSq <= rxSq * rySq) 1f else 0f
            }
        }
        return mask
    }

    private fun loadModelFile(context: Context, modelPath: String): MappedByteBuffer {
        val assetFd = context.assets.openFd(modelPath)
        return FileInputStream(assetFd.fileDescriptor).channel.map(
            FileChannel.MapMode.READ_ONLY,
            assetFd.startOffset,
            assetFd.declaredLength
        )
    }

    fun close() {
        engine.close()
        interpreter = null
    }
}
