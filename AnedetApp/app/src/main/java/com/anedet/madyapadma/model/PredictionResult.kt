package com.anedet.madyapadma.model

import android.graphics.Bitmap
import android.graphics.PointF
import android.graphics.RectF

/**
 * Hasil analisis anemia.
 *
 * - isAnemic: kelas diagnostik yang dipilih (anemic / non-anemic)
 * - anemicProbability: probabilitas kelas anemic (0..1)
 * - nonAnemicProbability: probabilitas kelas non-anemic (0..1)
 * - confidence: nilai keyakinan = max(anemicProbability, nonAnemicProbability)
 * - margin: selisih |anemicProbability - nonAnemicProbability| (keyakinan relatif)
 * - diagnosisClass: teks label diagnosis
 * - diagnosisPercent: probabilitas kelas yang dipilih (sama dengan confidence)
 *
 * confidence TIDAK sama dengan `anemicProbability` — confidence adalah nilai
 * maksimum antar kelas, sehingga confidence >= 50% selalu.
 */
data class PredictionResult(
    val isAnemic: Boolean,
    val anemicProbability: Float,
    val nonAnemicProbability: Float,
    val maskOverlay: Bitmap?,
    val inferenceTimeMs: Long,
    val error: String? = null,
    val bbox: RectF? = null,
    val polygon: List<PointF> = emptyList(),
    val croppedPreview: Bitmap? = null
) {
    /** Nilai keyakinan = probabilitas kelas pemenang. */
    val confidence: Float
        get() = maxOf(anemicProbability, nonAnemicProbability)

    /** Selisih absolut antara kedua kelas (0..1). */
    val margin: Float
        get() = kotlin.math.abs(anemicProbability - nonAnemicProbability)

    /** Probabilitas kelas yang menjadi diagnosis. */
    val diagnosisPercent: Float
        get() = if (isAnemic) anemicProbability else nonAnemicProbability
}
