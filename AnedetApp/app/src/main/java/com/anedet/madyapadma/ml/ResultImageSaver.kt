package com.anedet.madyapadma.ml

import android.content.ContentValues
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.os.Build
import android.provider.MediaStore
import com.anedet.madyapadma.model.PredictionResult
import java.io.OutputStream

object ResultImageSaver {

    fun saveResultImage(context: Context, original: Bitmap, result: PredictionResult): Boolean {
        val bitmap = original.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(bitmap)

        // 1. Draw filled ellipse (semi-transparent green) for the conjunctiva region
        result.bbox?.let { bbox ->
            val fillPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
                color = Color.argb(140, 76, 175, 80)
                style = Paint.Style.FILL
            }
            val cx = bbox.centerX()
            val cy = bbox.centerY()
            val rx = bbox.width()  / 2f
            val ry = bbox.height() / 2f
            canvas.drawOval(cx - rx, cy - ry, cx + rx, cy + ry, fillPaint)
        }

        // 2. Draw BBox outline
        val bboxPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = if (result.isAnemic) Color.RED else Color.GREEN
            style = Paint.Style.STROKE
            strokeWidth = 10f
        }
        result.bbox?.let { canvas.drawRect(it, bboxPaint) }

        // 3. Draw label + confidence
        val textPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.WHITE
            textSize = 64f
            isFakeBoldText = true
            setShadowLayer(5f, 1f, 1f, Color.BLACK)
        }
        val label = if (result.isAnemic)
            "ANEMIC (${"%.1f".format(result.anemicProbability * 100)}%)"
        else
            "NON-ANEMIC (${"%.1f".format(result.nonAnemicProbability * 100)}%)"

        val padding = 16f
        canvas.drawText(label, padding, 80f, textPaint)

        val infoPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.WHITE
            textSize = 40f
            setShadowLayer(3f, 1f, 1f, Color.BLACK)
        }
        canvas.drawText(
            "Confidence: ${"%.1f".format(result.confidence * 100)}%   Margin: ${"%.1f".format(result.margin * 100)}%",
            padding, 130f, infoPaint
        )

        // 4. Save to MediaStore
        return saveToGallery(context, bitmap)
    }

    private fun saveToGallery(context: Context, bitmap: Bitmap): Boolean {
        val filename = "ANEMIA_${System.currentTimeMillis()}.jpg"
        val contentValues = ContentValues().apply {
            put(MediaStore.MediaColumns.DISPLAY_NAME, filename)
            put(MediaStore.MediaColumns.MIME_TYPE, "image/jpeg")
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                put(MediaStore.MediaColumns.RELATIVE_PATH, "Pictures/AnemiaDetection")
            }
        }

        val uri = context.contentResolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, contentValues)
        return uri?.let {
            val outputStream: OutputStream? = context.contentResolver.openOutputStream(it)
            outputStream?.use { stream ->
                bitmap.compress(Bitmap.CompressFormat.JPEG, 92, stream)
            }
            true
        } ?: false
    }
}
