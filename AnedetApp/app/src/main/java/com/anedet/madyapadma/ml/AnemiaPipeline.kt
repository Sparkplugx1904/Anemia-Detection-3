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
import android.graphics.Path
import android.graphics.PorterDuff
import android.graphics.PorterDuffXfermode
import android.graphics.RectF
import com.anedet.madyapadma.model.MaskData
import com.anedet.madyapadma.model.PredictionResult
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlin.math.max

class AnemiaPipeline(
    private val context: Context,
    private val segmentor: Segmentor
) {

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

            // 2. Crop konjungtiva menggunakan Polygon/Mask Trace (Precision Crop)
            val croppedBitmap = cropByMask(original, segResult)
            
            if (croppedBitmap == null) {
                original.recycle()
                return@withContext PredictionResult(
                    isAnemic = false,
                    anemicProbability = 0f,
                    nonAnemicProbability = 0f,
                    maskOverlay = null,
                    inferenceTimeMs = elapsed(startTime),
                    error = "Gagal crop polygon konjungtiva"
                )
            }

            // 3. Klasifikasi — terima Bitmap hasil precision crop
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
                bbox = segResult.bbox,
                polygon = segResult.polygon
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
     * Precision Crop menggunakan Mask/Polygon.
     * Menggunakan PorterDuff Xfermode untuk nge-crop bitmap mengikuti trace mask.
     * Hasilnya adalah bitmap dimana hanya area konjungtiva yang punya pixel, sisanya hitam.
     */
    private fun cropByMask(original: Bitmap, segResult: MaskData): Bitmap? {
        return try {
            val mask = segResult.mask
            val mh = mask.size
            val mw = if (mh > 0) mask[0].size else 0
            if (mw == 0) return null

            // 1. Buat bitmap mask binary seukuran proto (160x160)
            val protoMaskBmp = Bitmap.createBitmap(mw, mh, Bitmap.Config.ALPHA_8)
            val pixels = ByteArray(mw * mh)
            for (y in 0 until mh) {
                for (x in 0 until mw) {
                    // Binary threshold: 255 (visible) atau 0 (transparent)
                    pixels[y * mw + x] = if (mask[y][x] > 0.5f) 255.toByte() else 0.toByte()
                }
            }
            protoMaskBmp.copyPixelsFromBuffer(java.nio.ByteBuffer.wrap(pixels))

            // 2. Siapkan canvas seukuran original untuk proses masking
            val result = Bitmap.createBitmap(original.width, original.height, Bitmap.Config.ARGB_8888)
            val canvas = Canvas(result)
            val paint = Paint(Paint.ANTI_ALIAS_FLAG)

            // 3. Gambar mask yang sudah di-scale ke koordinat original
            val matrix = buildMaskMatrix(
                mw, mh, segResult.protoW, segResult.protoH,
                original.width, original.height,
                segResult.lbScale, segResult.lbPadLeft, segResult.lbPadTop
            )
            canvas.drawBitmap(protoMaskBmp, matrix, paint)
            protoMaskBmp.recycle()

            // 4. SRC_IN: Ambil area "original" yang overlap dengan "mask"
            paint.xfermode = PorterDuffXfermode(PorterDuff.Mode.SRC_IN)
            canvas.drawBitmap(original, 0f, 0f, paint)
            paint.xfermode = null

            // 5. Crop hasil akhirnya ke bounding box polygon (lebih ketat dari YOLO bbox)
            //    Fallback ke YOLO bbox kalau polygon kosong.
            val cropBox = if (segResult.polygon.size >= 3) {
                computePolygonBbox(segResult.polygon)
            } else {
                segResult.bbox
            }
            cropConjunctiva(result, cropBox)
        } catch (e: Exception) {
            null
        }
    }

    /**
     * Hitung bounding box dari daftar titik polygon.
     * Titik polygon sudah dalam koordinat gambar original.
     */
    private fun computePolygonBbox(polygon: List<android.graphics.PointF>): RectF {
        if (polygon.isEmpty()) return RectF()
        val minX = polygon.minOf { it.x }
        val minY = polygon.minOf { it.y }
        val maxX = polygon.maxOf { it.x }
        val maxY = polygon.maxOf { it.y }
        return RectF(minX, minY, maxX, maxY)
    }

    /**
     * Crop region bbox dari bitmap original. Tidak melakukan resize di sini —
     * Classifier akan menangani letterbox ke ukuran inputnya sendiri.
     * Menghindari double-resize yang merusak proporsi konjungtiva.
     */
    private fun cropConjunctiva(bitmap: Bitmap, bbox: RectF): Bitmap? {
        return try {
            val left   = bbox.left.coerceIn(0f, (bitmap.width  - 1).toFloat())
            val top    = bbox.top.coerceIn(0f,  (bitmap.height - 1).toFloat())
            val right  = bbox.right.coerceIn(left + 1f, bitmap.width.toFloat())
            val bottom = bbox.bottom.coerceIn(top + 1f, bitmap.height.toFloat())

            val cropW = (right - left).toInt().coerceAtLeast(1)
            val cropH = (bottom - top).toInt().coerceAtLeast(1)

            Bitmap.createBitmap(bitmap, left.toInt(), top.toInt(), cropW, cropH)
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
     * Hasilnya: Bitmap ARGB_8888 seukuran original dengan overlay hijau semi-transparan
     * + outline polygon konjungtiva.
     */
    private fun createMaskOverlay(original: Bitmap, segResult: MaskData): Bitmap? {
        return try {
            val mask = segResult.mask
            val maskH = mask.size
            val maskW = if (maskH > 0) mask[0].size else 0
            if (maskW == 0) return null

            // 1. Buat overlay seukuran original
            val overlay = original.copy(Bitmap.Config.ARGB_8888, true)
            val canvas = Canvas(overlay)

            // 2. Buat bitmap mask transparan (proto space / mask space)
            val maskBitmap = Bitmap.createBitmap(maskW, maskH, Bitmap.Config.ARGB_8888)
            val maskPixels = IntArray(maskW * maskH)
            for (y in 0 until maskH) {
                for (x in 0 until maskW) {
                    val alpha = if (mask[y][x] > 0.5f) 160 else 0
                    maskPixels[y * maskW + x] = Color.argb(alpha, 76, 175, 80)
                }
            }
            maskBitmap.setPixels(maskPixels, 0, maskW, 0, 0, maskW, maskH)

            // 3. Transform mask space -> original image space
            val matrix = if (segResult.protoW > 0) {
                buildMaskMatrix(
                    maskW, maskH,
                    segResult.protoW, segResult.protoH,
                    original.width, original.height,
                    segResult.lbScale, segResult.lbPadLeft, segResult.lbPadTop
                )
            } else {
                // Fallback: scale mask langsung ke original
                Matrix().apply {
                    postScale(original.width.toFloat() / maskW, original.height.toFloat() / maskH)
                }
            }

            canvas.drawBitmap(maskBitmap, matrix, Paint(Paint.FILTER_BITMAP_FLAG))
            maskBitmap.recycle()

            // 4. Gambar outline polygon konjungtiva (bukan bbox rectangle).
            //    Titik polygon sudah dalam koordinat gambar original.
            if (segResult.polygon.size >= 3) {
                val polyPath = polygonToPath(segResult.polygon)
                val polyPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
                    color = Color.YELLOW
                    style = Paint.Style.STROKE
                    strokeWidth = max(3f, original.width / 350f)
                }
                canvas.drawPath(polyPath, polyPaint)
            }

            overlay
        } catch (e: Exception) {
            null
        }
    }

    /**
     * Konversi daftar titik polygon menjadi android.graphics.Path tertutup.
     */
    private fun polygonToPath(points: List<android.graphics.PointF>): Path {
        val path = Path()
        if (points.isEmpty()) return path
        path.moveTo(points[0].x, points[0].y)
        for (i in 1 until points.size) {
            path.lineTo(points[i].x, points[i].y)
        }
        path.close()
        return path
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
