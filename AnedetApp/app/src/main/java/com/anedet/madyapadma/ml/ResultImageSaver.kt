package com.anedet.madyapadma.ml

import android.content.ContentValues
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.os.Build
import android.provider.MediaStore
import com.anedet.madyapadma.model.PredictionResult
import java.io.OutputStream

object ResultImageSaver {

    fun saveResultImage(context: Context, original: Bitmap, result: PredictionResult): Boolean {
        val bitmap = original.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(bitmap)

        // 1. Draw mask
        result.maskOverlay?.let { mask ->
            val paint = Paint().apply { alpha = 128 }
            canvas.drawBitmap(mask, 0f, 0f, paint)
        }

        // 2. Draw BBox and Label
        val paint = Paint().apply {
            color = if (result.isAnemic) Color.RED else Color.GREEN
            style = Paint.Style.STROKE
            strokeWidth = 8f
        }

        result.bbox?.let { canvas.drawRect(it, paint) }

        val textPaint = Paint().apply {
            color = Color.WHITE
            textSize = 64f
            isFakeBoldText = true
            setShadowLayer(5f, 0f, 0f, Color.BLACK)
        }

        val label = if (result.isAnemic) "ANEMIC (${(result.anemicProbability*100).toInt()}%)"
                    else "NON-ANEMIC (${(result.nonAnemicProbability*100).toInt()}%)"

        canvas.drawText(label, 50f, 100f, textPaint)

        // 3. Save to MediaStore
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
                bitmap.compress(Bitmap.CompressFormat.JPEG, 90, stream)
            }
            true
        } ?: false
    }
}
