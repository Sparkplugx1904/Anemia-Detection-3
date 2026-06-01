package com.anedet.madyapadma.model

import android.graphics.Bitmap
import android.graphics.RectF

data class PredictionResult(
    val isAnemic: Boolean,
    val anemicProbability: Float,
    val nonAnemicProbability: Float,
    val maskOverlay: Bitmap?,
    val inferenceTimeMs: Long,
    val error: String? = null,
    val bbox: RectF? = null
) {
    val confidence: Float
        get() = if (isAnemic) anemicProbability else nonAnemicProbability
}
