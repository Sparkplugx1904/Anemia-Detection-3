package com.anedet.madyapadma.ml

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.ColorFilter
import android.graphics.ColorMatrix
import android.graphics.ColorMatrixColorFilter
import android.graphics.LightingColorFilter
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.PorterDuff
import android.graphics.PorterDuffXfermode
import android.graphics.RectF
import com.anedet.madyapadma.model.MaskData
import com.anedet.madyapadma.model.PredictionResult
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlin.math.max

class AnemiaPipeline(private val context: Context) {

    // Init di konstruktor, bukan lazy — hindari race condition
    private val segmentor = Segmentor(context)
    private val classifier = Classifier(context)

    suspend fun initialize() {
        segmentor.initialize()
        classifier.initialize()
    }

    suspend fun analyze(imagePath: String): PredictionResult = withContext(Dispatchers.Default) {
        val startTime = System.currentTimeMillis()

        try {
            // 1. Segmentasi
            val segResult = segmentor.runSegmentation(imagePath)
                ?: return@withContext PredictionResult(
                    isAnemic = false,
                    anemicProbability = 0f,
                    nonAnemicProbability = 0f,
                    maskOverlay = null,
                    inferenceTimeMs = elapsed(startTime),
                    error = "Konjungtiva tidak terdeteksi"
                )

            // 2. Crop konjungtiva langsung sebagai Bitmap (tanpa roundtrip ke disk)
            val original = BitmapFactory.decodeFile(imagePath)
                ?: return@withContext PredictionResult(
                    isAnemic = false,
                    anemicProbability = 0f,
                    nonAnemicProbability = 0f,
                    maskOverlay = null,
                    inferenceTimeMs = elapsed(startTime),
                    error = "Gagal decode gambar"
                )

            val croppedBitmap = cropConjunctiva(original, segResult.bbox)
            if (croppedBitmap == null) {
                original.recycle()
                return@withContext PredictionResult(
                    isAnemic = false,
                    anemicProbability = 0f,
                    nonAnemicProbability = 0f,
                    maskOverlay = null,
                    inferenceTimeMs = elapsed(startTime),
                    error = "Gagal crop konjungtiva"
                )
            }

            // 3. Klasifikasi — terima Bitmap langsung
            val clsResult = classifier.classify(croppedBitmap)
            croppedBitmap.recycle()

            if (clsResult == null) {
                original.recycle()
                return@withContext PredictionResult(
                    isAnemic = false,
                    anemicProbability = 0f,
                    nonAnemicProbability = 0f,
                    maskOverlay = null,
                    inferenceTimeMs = elapsed(startTime),
                    error = "Klasifikasi gagal"
                )
            }

            // 4. Buat mask overlay dari proto space (efisien)
            val maskOverlay = createMaskOverlay(original, segResult)
            original.recycle()

            val (anemicProb, nonAnemicProb) = clsResult

            PredictionResult(
                isAnemic = anemicProb > nonAnemicProb,
                anemicProbability = anemicProb,
                nonAnemicProbability = nonAnemicProb,
                maskOverlay = maskOverlay,
                inferenceTimeMs = elapsed(startTime),
                bbox = segResult.bbox
            )

        } catch (e: Exception) {
            PredictionResult(
                isAnemic = false,
                anemicProbability = 0f,
                nonAnemicProbability = 0f,
                maskOverlay = null,
                inferenceTimeMs = elapsed(startTime),
                error = e.message ?: "Unknown error"
            )
        }
    }

    /**
     * Crop region bbox dari bitmap original dan resize ke ukuran input classifier.
     * Tidak menulis ke disk — langsung kembalikan Bitmap.
     */
    private fun cropConjunctiva(bitmap: Bitmap, bbox: RectF): Bitmap? {
        return try {
            val left   = bbox.left.coerceIn(0f, (bitmap.width  - 1).toFloat())
            val top    = bbox.top.coerceIn(0f,  (bitmap.height - 1).toFloat())
            val right  = bbox.right.coerceIn(left + 1f, bitmap.width.toFloat())
            val bottom = bbox.bottom.coerceIn(top + 1f, bitmap.height.toFloat())

            val cropW = (right - left).toInt().coerceAtLeast(1)
            val cropH = (bottom - top).toInt().coerceAtLeast(1)

            val cropped = Bitmap.createBitmap(bitmap, left.toInt(), top.toInt(), cropW, cropH)
            // Resize ke input size classifier — classifier akan handle letterbox sendiri
            val resized = Bitmap.createScaledBitmap(cropped, Classifier.INPUT_SIZE, Classifier.INPUT_SIZE, true)
            if (cropped !== resized) cropped.recycle()
            resized
        } catch (e: Exception) {
            null
        }
    }

    /**
     * Buat overlay bitmap dari mask MaskData.
     *
     * Jika mask ada di proto space (isProtoSpace = true):
     *   - Buat Bitmap kecil dari proto mask (160×160)
     *   - Scale ke ukuran display menggunakan Matrix + Canvas — satu operasi GPU-accelerated
     *   - Jauh lebih cepat dari per-pixel drawPoint()
     *
     * Hasilnya: Bitmap ARGB_8888 seukuran original dengan overlay hijau semi-transparan.
     */
    private fun createMaskOverlay(original: Bitmap, segResult: MaskData): Bitmap? {
        return try {
            val mask = segResult.mask
            val maskH = mask.size
            val maskW = if (maskH > 0) mask[0].size else 0
            if (maskW == 0) return null

            // 1. Buat overlay bitmap seukuran original (langsung gambar di sini)
            val overlay = original.copy(Bitmap.Config.ARGB_8888, true)
            val canvas = Canvas(overlay)

            // 2. Gambar filled ellipse mask (semi-transparan hijau) langsung pada overlay
            val fillPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
                color = Color.argb(140, 76, 175, 80)
                style = Paint.Style.FILL
            }
            val bbox = segResult.bbox
            val cx = bbox.centerX()
            val cy = bbox.centerY()
            val rx = bbox.width()  / 2f
            val ry = bbox.height() / 2f
            canvas.drawOval(cx - rx, cy - ry, cx + rx, cy + ry, fillPaint)

            // 3. Gambar stroke ellipse agar lebih jelas
            val strokePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
                color = Color.argb(255, 27, 94, 32)   // dark green stroke
                style = Paint.Style.STROKE
                strokeWidth = max(4f, original.width / 250f)
            }
            canvas.drawOval(cx - rx, cy - ry, cx + rx, cy + ry, strokePaint)

            // 4. Gambar bbox rectangle dengan warna diagnosis
            val bboxPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
                color = Color.YELLOW
                style = Paint.Style.STROKE
                strokeWidth = max(3f, original.width / 350f)
            }
            canvas.drawRect(bbox, bboxPaint)

            overlay
        } catch (e: Exception) {
            null
        }
    }

    /**
     * Bangun Matrix untuk memetakan mask dari proto space ke koordinat gambar original.
     *
     * Alur koordinat:
     *   proto px → input model (INPUT_SIZE=320) → gambar original (kompensasi letterbox)
     *
     *   protoToModel : scale = INPUT_SIZE / protoW (dan INPUT_SIZE / protoH)
     *   modelToOrig  : (coord - pad) / lbScale
     *
     * Jadi protoToOrig:
     *   origX = (protoX * INPUT_SIZE/protoW - padLeft) / lbScale
     *   origX = protoX * (INPUT_SIZE / protoW / lbScale) - padLeft / lbScale
     */
    private fun buildMaskMatrix(
        maskW: Int, maskH: Int,
        protoW: Int, protoH: Int,
        imgW: Int, imgH: Int,
        lbScale: Float,
        lbPadLeft: Int, lbPadTop: Int
    ): Matrix {
        val modelPerProtoX = Segmentor.INPUT_SIZE.toFloat() / protoW
        val modelPerProtoY = Segmentor.INPUT_SIZE.toFloat() / protoH

        val scaleX = modelPerProtoX / lbScale
        val scaleY = modelPerProtoY / lbScale
        val transX = -lbPadLeft.toFloat() / lbScale
        val transY = -lbPadTop.toFloat()  / lbScale

        return Matrix().apply {
            // Scale dari mask space ke proto space (jika mask == proto, ini 1x)
            postScale(protoW.toFloat() / maskW, protoH.toFloat() / maskH)
            // Scale + translate proto → original image
            postScale(scaleX, scaleY)
            postTranslate(transX, transY)
        }
    }

    private fun elapsed(startTime: Long) = System.currentTimeMillis() - startTime

    fun close() {
        segmentor.close()
        classifier.close()
    }
}
