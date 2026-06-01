package com.anedet.madyapadma.model

import android.graphics.RectF

data class MaskData(
    val bbox: RectF,
    val mask: Array<FloatArray>,
    val confidence: Float
)
