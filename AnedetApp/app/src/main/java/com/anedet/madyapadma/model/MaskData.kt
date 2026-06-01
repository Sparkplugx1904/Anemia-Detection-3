package com.anedet.madyapadma.model

import android.graphics.RectF

data class MaskData(
    val bbox: RectF,
    val mask: Array<FloatArray>,
    val confidence: Float,
    val protoW: Int = 0,
    val protoH: Int = 0,
    val lbScale: Float = 1f,
    val lbPadLeft: Int = 0,
    val lbPadTop: Int = 0
) {
    val isProtoSpace: Boolean get() = protoW > 0 && protoH > 0

    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is MaskData) return false
        return confidence == other.confidence &&
               protoW == other.protoW && protoH == other.protoH &&
               bbox == other.bbox
    }

    override fun hashCode(): Int = 31 * bbox.hashCode() + confidence.hashCode()
}
