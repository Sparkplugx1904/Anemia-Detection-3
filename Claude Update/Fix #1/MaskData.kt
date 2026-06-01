package com.anedet.madyapadma.model

import android.graphics.RectF

/**
 * Hasil segmentasi dari Segmentor.
 *
 * [bbox]        : Bounding box dalam koordinat gambar original (pixel).
 * [mask]        : Mask biner. Jika [isProtoSpace] = true, ukurannya [protoW]×[protoH]
 *                 dan perlu di-upsample saat render. Jika false, ukurannya imgW×imgH.
 * [confidence]  : Confidence score deteksi.
 * [protoW/H]    : Dimensi proto space (biasanya 160×160). Hanya valid jika [isProtoSpace] = true.
 * [lbScale]     : Scale letterbox yang digunakan saat preprocessing.
 * [lbPadLeft/Top]: Offset padding letterbox (px dalam input model 320×320).
 */
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
    /** True jika mask ada di proto space (perlu upsample ke display size). */
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
