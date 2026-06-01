package com.anedet.madyapadma.ml

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import com.anedet.madyapadma.model.PredictionResult
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

class AnemiaPipeline(private val context: Context) {

    private var segmentor: Segmentor? = null
    private var classifier: Classifier? = null

    suspend fun analyze(imagePath: String): PredictionResult = withContext(Dispatchers.Default) {
        val startTime = System.currentTimeMillis()

        try {
            ensureModels()

            val segResult = segmentor?.runSegmentation(imagePath)
            if (segResult == null) {
                return@withContext PredictionResult(
                    isAnemic = false,
                    anemicProbability = 0f,
                    nonAnemicProbability = 0f,
                    maskOverlay = null,
                    inferenceTimeMs = System.currentTimeMillis() - startTime,
                    error = "No conjunctiva detected"
                )
            }

            val croppedPath = cropConjunctiva(imagePath, segResult.bbox)
            if (croppedPath == null) {
                return@withContext PredictionResult(
                    isAnemic = false,
                    anemicProbability = 0f,
                    nonAnemicProbability = 0f,
                    maskOverlay = null,
                    inferenceTimeMs = System.currentTimeMillis() - startTime,
                    error = "Failed to crop conjunctiva"
                )
            }

            val clsResult = classifier?.classify(croppedPath)
            if (clsResult == null) {
                return@withContext PredictionResult(
                    isAnemic = false,
                    anemicProbability = 0f,
                    nonAnemicProbability = 0f,
                    maskOverlay = null,
                    inferenceTimeMs = System.currentTimeMillis() - startTime,
                    error = "Classification failed"
                )
            }

            val (anemicProb, nonAnemicProb) = clsResult
            val maskOverlay = createMaskOverlay(imagePath, segResult.mask)
            val elapsed = System.currentTimeMillis() - startTime

            PredictionResult(
                isAnemic = anemicProb > nonAnemicProb,
                anemicProbability = anemicProb,
                nonAnemicProbability = nonAnemicProb,
                maskOverlay = maskOverlay,
                inferenceTimeMs = elapsed
            )
        } catch (e: Exception) {
            PredictionResult(
                isAnemic = false,
                anemicProbability = 0f,
                nonAnemicProbability = 0f,
                maskOverlay = null,
                inferenceTimeMs = System.currentTimeMillis() - startTime,
                error = e.message ?: "Unknown error"
            )
        }
    }

    private fun ensureModels() {
        if (segmentor == null) segmentor = Segmentor(context)
        if (classifier == null) classifier = Classifier(context)
    }

    private fun cropConjunctiva(imagePath: String, bbox: RectF): String? {
        try {
            val bitmap = BitmapFactory.decodeFile(imagePath) ?: return null

            val left = bbox.left.coerceAtLeast(0f)
            val top = bbox.top.coerceAtLeast(0f)
            val right = bbox.right.coerceAtMost(bitmap.width.toFloat())
            val bottom = bbox.bottom.coerceAtMost(bitmap.height.toFloat())
            val width = (right - left).toInt().coerceAtLeast(1)
            val height = (bottom - top).toInt().coerceAtLeast(1)

            val cropped = Bitmap.createBitmap(bitmap, left.toInt(), top.toInt(), width, height)
            val resized = Bitmap.createScaledBitmap(cropped, Classifier.INPUT_SIZE, Classifier.INPUT_SIZE, true)

            val cropFile = java.io.File(context.cacheDir, "crop_${System.currentTimeMillis()}.jpg")
            resized.compress(Bitmap.CompressFormat.JPEG, 90, java.io.FileOutputStream(cropFile))

            cropped.recycle()
            resized.recycle()
            bitmap.recycle()

            return cropFile.absolutePath
        } catch (e: Exception) {
            return null
        }
    }

    private fun createMaskOverlay(imagePath: String, mask: Array<FloatArray>): Bitmap? {
        try {
            val bitmap = BitmapFactory.decodeFile(imagePath) ?: return null
            val overlay = bitmap.copy(Bitmap.Config.ARGB_8888, true)
            val canvas = Canvas(overlay)

            val maskPaint = Paint().apply {
                color = Color.argb(100, 76, 175, 80)
                style = Paint.Style.FILL
            }
            val borderPaint = Paint().apply {
                color = Color.argb(200, 76, 175, 80)
                style = Paint.Style.STROKE
                strokeWidth = 4f
            }

            val scaleY = mask.size.toFloat() / overlay.height
            val scaleX = mask[0].size.toFloat() / overlay.width

            for (y in 0 until overlay.height) {
                for (x in 0 until overlay.width) {
                    val my = (y * scaleY).toInt().coerceAtMost(mask.size - 1)
                    val mx = (x * scaleX).toInt().coerceAtMost(mask[0].size - 1)
                    if (mask[my][mx] > 0.5f) {
                        canvas.drawPoint(x.toFloat(), y.toFloat(), maskPaint)
                    }
                }
            }

            bitmap.recycle()
            return overlay
        } catch (e: Exception) {
            return null
        }
    }

    fun close() {
        segmentor?.close()
        classifier?.close()
    }
}
