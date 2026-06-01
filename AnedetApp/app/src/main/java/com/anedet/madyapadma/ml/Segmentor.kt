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
import kotlin.math.exp
import kotlin.math.min

class Segmentor(context: Context) {

    companion object {
        private const val MODEL_PATH = "yolo26n_seg_fp16.tflite"
        const val INPUT_SIZE = 320
        private const val CONF_THRESHOLD = 0.25f
        // Nilai fill letterbox (0.5 = gray, sesuai YOLO training default)
        private const val LETTERBOX_FILL = 0.5f
    }

    private var bundle: InterpreterBundle? = null
    private var outputShapes = listOf<IntArray>()

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

    init {
        loadModel(context)
    }

    private fun loadModel(context: Context) {
        val modelBuffer = loadModelFile(context, MODEL_PATH)
        bundle = TfLiteHelper.createInterpreter(modelBuffer)
        val interp = bundle?.interpreter ?: throw RuntimeException("Failed to create interpreter")

        val numOutputs = interp.outputTensorCount
        outputShapes = (0 until numOutputs).map { idx ->
            interp.getOutputTensor(idx)?.shape() ?: intArrayOf()
        }
    }

    fun runSegmentation(imagePath: String): MaskData? {
        val bitmap = BitmapFactory.decodeFile(imagePath) ?: return null
        val origW = bitmap.width
        val origH = bitmap.height

        val lbParams = preprocess(bitmap)
        bitmap.recycle()

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
        val interp = bundle?.interpreter ?: return null
        val numOutputs = outputShapes.size
        if (numOutputs == 0) return null

        val outputs = arrayOfNulls<Any>(numOutputs)
        for (i in 0 until numOutputs) {
            val shape = outputShapes[i]
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
                @Suppress("UNCHECKED_CAST")
                (out as Array<Any>)[0] as Array<Any>
            } else {
                out as Array<Any>
            }
        }
    }

    /**
     * Parse output YOLO-seg:
     *   outputs[0]: detection tensor, shape setelah strip batch → [detRows, detCols] atau [numCols, numRows]
     *   outputs[1]: proto tensor, shape setelah strip batch    → [protoH, protoW, numProtoChannels]
     *
     * Koordinat: model output x1y1x2y2 dalam skala INPUT_SIZE (320px).
     * Perlu di-decode balik ke koordinat gambar original dengan memperhitungkan letterbox.
     */
    private fun parseOutput(outputs: List<Array<Any>>, lbParams: LetterboxParams): MaskData? {
        if (outputs.isEmpty()) return null

        // --- Baca shape detection output ---
        // Shape biasanya [numDetections, 4+1+numMaskCoeffs] atau transposed
        val detShape = outputShapes[0]

        // Deteksi apakah transposed: [cols, rows] vs [rows, cols]
        // YOLO biasanya output [1, numCols, numRows] dimana numRows = 4+conf+coeffs
        // Setelah strip batch: [numCols, numRows]
        val det0 = outputs[0]

        // Cek orientasi: baris mana yang berisi konfidence?
        // Jika det[4][col] → berarti dim0=rows, dim1=cols (layout: row=feature, col=anchor)
        // Jika det[col][4] → berarti dim0=anchors, dim1=features
        val isRowMajor = run {
            // Perkiraan: jika dim0 kecil (<=10), kemungkinan itu adalah feature rows
            val d = detShape.drop(1) // sudah strip batch
            d.isNotEmpty() && d[0] <= d.getOrElse(1) { 0 }
        }

        // Ambil jumlah anchor dan feature size
        val numAnchors: Int
        val numFeatures: Int

        if (isRowMajor) {
            // det[featureIdx][anchorIdx]
            numFeatures = detShape.getOrElse(1) { 0 }
            numAnchors  = detShape.getOrElse(2) { 0 }
        } else {
            // det[anchorIdx][featureIdx]
            numAnchors  = detShape.getOrElse(1) { 0 }
            numFeatures = detShape.getOrElse(2) { 0 }
        }

        if (numFeatures < 5 || numAnchors <= 0) return null

        // Cari deteksi dengan confidence tertinggi
        var bestConf = 0f
        var bestIdx  = -1

        for (ancIdx in 0 until numAnchors) {
            val conf = if (isRowMajor) {
                (det0[4] as FloatArray)[ancIdx]
            } else {
                (det0[ancIdx] as FloatArray)[4]
            }
            if (conf > bestConf && conf > CONF_THRESHOLD) {
                bestConf = conf
                bestIdx  = ancIdx
            }
        }

        if (bestIdx < 0) return null

        // Ambil koordinat x1y1x2y2 dalam skala INPUT_SIZE
        fun getFeature(featureIdx: Int): Float {
            return if (isRowMajor) {
                (det0[featureIdx] as FloatArray)[bestIdx]
            } else {
                (det0[bestIdx] as FloatArray)[featureIdx]
            }
        }

        val x1Model = getFeature(0)
        val y1Model = getFeature(1)
        val x2Model = getFeature(2)
        val y2Model = getFeature(3)

        // Decode dari skala model ke koordinat gambar original (kompensasi letterbox)
        val bbox = decodeBbox(x1Model, y1Model, x2Model, y2Model, lbParams)

        // Kalau tidak ada output proto, kembalikan fallback mask
        if (outputs.size < 2) {
            return MaskData(bbox, fallbackMask(bbox, lbParams.origW, lbParams.origH), bestConf)
        }

        // --- Baca proto tensor ---
        // Shape setelah strip batch: [protoH, protoW, numProtoChannels]
        val protoShape = outputShapes[1]
        if (protoShape.size < 4) return MaskData(bbox, fallbackMask(bbox, lbParams.origW, lbParams.origH), bestConf)

        val protoH = protoShape[1]
        val protoW = protoShape[2]
        val numProtoChannels = protoShape[3]

        // Ambil mask coefficients
        val numCoeffs = numFeatures - 5
        val maskCoeffs = FloatArray(minOf(numCoeffs, numProtoChannels)) { c ->
            getFeature(5 + c)
        }

        // Rekonstruksi mask di proto space (protoH×protoW), BUKAN di image space
        // Ini jauh lebih cepat: 160×160×32 vs 720×1280×32
        @Suppress("UNCHECKED_CAST")
        val protoTensor = outputs[1] as Array<Array<FloatArray>>
        // protoTensor[y][x] = FloatArray(numProtoChannels)

        val maskProto = Array(protoH) { py ->
            FloatArray(protoW) { px ->
                var sum = 0f
                for (c in maskCoeffs.indices) {
                    sum += maskCoeffs[c] * protoTensor[py][px][c]
                }
                // Sigmoid activation
                if (sum >= 0) 1f / (1f + exp(-sum)) else {
                    val e = exp(sum); e / (1f + e)
                }
            }
        }

        // Crop mask proto ke region bbox (dalam skala proto)
        // Konversi bbox model → proto coords
        val px1 = ((x1Model / INPUT_SIZE) * protoW).toInt().coerceIn(0, protoW - 1)
        val py1 = ((y1Model / INPUT_SIZE) * protoH).toInt().coerceIn(0, protoH - 1)
        val px2 = ((x2Model / INPUT_SIZE) * protoW).toInt().coerceIn(0, protoW - 1)
        val py2 = ((y2Model / INPUT_SIZE) * protoH).toInt().coerceIn(0, protoH - 1)

        // Apply threshold di proto space
        for (py in 0 until protoH) {
            for (px in 0 until protoW) {
                // Zero-out di luar bbox region
                if (py < py1 || py > py2 || px < px1 || px > px2) {
                    maskProto[py][px] = 0f
                } else {
                    maskProto[py][px] = if (maskProto[py][px] > 0.5f) 1f else 0f
                }
            }
        }

        return MaskData(bbox, maskProto, bestConf, protoW, protoH, lbParams.scale, lbParams.padLeft, lbParams.padTop)
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
        bundle?.close()
        bundle = null
    }
}
