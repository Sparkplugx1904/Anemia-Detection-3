package com.anedet.madyapadma.camera

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.PorterDuff
import android.graphics.Rect
import com.anedet.madyapadma.ml.Segmentor
import com.anedet.madyapadma.model.MaskData
import kotlin.math.sqrt

/**
 * Utility for image analysis and quality checks.
 */
object ImageQualityUtils {

    /**
     * Quick pre-check apakah image worth untuk di-inference.
     * Menghindari wasted inference pada low-quality frames.
     * 
     * @param bitmap Input image
     * @param minSharpness Minimum sharpness threshold (default from AppSettings)
     * @return true if image quality sufficient for inference
     */
    fun isQualitySufficientForInference(bitmap: Bitmap, minSharpness: Float = 100f): Boolean {
        // Quick dimension check - reject too small images
        if (bitmap.width < 160 || bitmap.height < 160) {
            return false
        }
        
        // Sharpness check - but use subsampled image untuk speed
        // (calculateBlurriness on full image bisa 30-50ms, subsample jadi ~10ms)
        val subsampledSharpness = if (bitmap.width > 640 || bitmap.height > 640) {
            val scale = 640f / maxOf(bitmap.width, bitmap.height)
            val smallW = (bitmap.width * scale).toInt()
            val smallH = (bitmap.height * scale).toInt()
            val small = Bitmap.createScaledBitmap(bitmap, smallW, smallH, false)
            val sharpness = calculateBlurriness(small)
            small.recycle()
            sharpness
        } else {
            calculateBlurriness(bitmap)
        }
        
        return subsampledSharpness >= minSharpness
    }

    /**
     * Calculates Laplacian Variance for sharpness.
     * Higher value = sharper.
     */
    fun calculateBlurriness(bitmap: Bitmap): Float {
        val w = bitmap.width
        val h = bitmap.height
        val pixels = IntArray(w * h)
        bitmap.getPixels(pixels, 0, w, 0, 0, w, h)

        val laplacian = IntArray(w * h)
        val kernel = intArrayOf(
            0,  1, 0,
            1, -4, 1,
            0,  1, 0
        )

        var mean = 0.0
        for (y in 1 until h - 1) {
            for (x in 1 until w - 1) {
                var sum = 0
                for (ky in -1..1) {
                    for (kx in -1..1) {
                        val p = pixels[(y + ky) * w + (x + kx)]
                        val lum = ((p shr 16 and 0xFF) * 0.299 + (p shr 8 and 0xFF) * 0.587 + (p and 0xFF) * 0.114).toInt()
                        sum += lum * kernel[(ky + 1) * 3 + (kx + 1)]
                    }
                }
                laplacian[y * w + x] = sum
                mean += sum
            }
        }
        mean /= (w * h)

        var variance = 0.0
        for (i in laplacian.indices) {
            variance += (laplacian[i] - mean) * (laplacian[i] - mean)
        }
        return (variance / laplacian.size).toFloat()
    }
}
