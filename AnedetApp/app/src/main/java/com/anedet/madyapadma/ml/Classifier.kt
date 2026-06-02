package com.anedet.madyapadma.ml

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import org.tensorflow.lite.InterpreterApi
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow
import kotlin.math.sqrt

/**
 * Classifier konjungtiva anemia.
 * Output: class 0 = Anemia, class 1 = Non-Anemia
 *
 * Preprocessing pipeline v2 (sesuai training):
 *   1. GrayWorldWB(strength=0.8)    — white balance koreksi warna
 *   2. AdaptiveGamma(0.5–1.2)       — normalisasi kecerahan
 *   3. Letterbox(224px)             — resize preserving aspect ratio
 *   4. BilateralFilter approx       — smooth noise, preserve edge
 *   5. AdaptiveCLAHE(clip 8–25)     — contrast enhancement di L* channel
 *   6. Normalize /255.0f            — ke [0,1]
 */
class Classifier(private val context: Context) {

    companion object {
        private const val MODEL_PATH = "yolo26s_cls_fp16.tflite"
        const val INPUT_SIZE = 448    // Letterbox target (model dilatih di 448px)
        private const val CLIP_MIN   = 8f
        private const val CLIP_MAX   = 25f
        private const val WB_STRENGTH = 0.8f
        private const val GAMMA_MIN  = 0.5f
        private const val GAMMA_MAX  = 1.2f
    }

    private var engine: TfLiteEngine = TfLiteEngine(context)
    private var interpreter: InterpreterApi? = null

    // Pre-allocated buffer
    private val inputBuffer: ByteBuffer = ByteBuffer
        .allocateDirect(1 * INPUT_SIZE * INPUT_SIZE * 3 * 4)
        .also { it.order(ByteOrder.nativeOrder()) }

    suspend fun initialize() {
        if (interpreter != null) return
        engine.initialize()
        val modelBuffer = loadModelFile(this.context, MODEL_PATH)
        interpreter = engine.createInterpreter(modelBuffer, useGpu = true)
    }

    /**
     * Klasifikasi dari Bitmap yang sudah di-crop dari region konjungtiva.
     * Mengembalikan Pair(anemicProb, nonAnemicProb).
     */
    suspend fun classify(bitmap: Bitmap): Pair<Float, Float>? {
        if (interpreter == null) initialize()
        return try {
            preprocessToBuffer(bitmap)
            runClassification()
        } catch (e: Exception) {
            null
        }
    }

    private fun preprocessToBuffer(src: Bitmap) {
        // Step 1: GrayWorldWB
        val wbBitmap = grayWorldWB(src, WB_STRENGTH)

        // Step 2: AdaptiveGamma
        val gammaBitmap = adaptiveGamma(wbBitmap, GAMMA_MIN, GAMMA_MAX)
        wbBitmap.recycle()

        // Step 3: Letterbox ke INPUT_SIZE
        val (lbBitmap, _, _) = letterbox(gammaBitmap, INPUT_SIZE)
        gammaBitmap.recycle()

        // Step 4: BilateralFilter approximation (3x3 range-weighted)
        val bilBitmap = bilateralFilterApprox(lbBitmap)
        lbBitmap.recycle()

        // Step 5: AdaptiveCLAHE di L* channel
        val claheBitmap = adaptiveCLAHE(bilBitmap, CLIP_MIN, CLIP_MAX)
        bilBitmap.recycle()

        // Step 6: Tulis ke ByteBuffer [R,G,B] / 255.0f
        inputBuffer.rewind()
        val pixels = IntArray(INPUT_SIZE * INPUT_SIZE)
        claheBitmap.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)
        claheBitmap.recycle()

        for (pixel in pixels) {
            inputBuffer.putFloat(((pixel shr 16) and 0xFF) / 255.0f)
            inputBuffer.putFloat(((pixel shr 8)  and 0xFF) / 255.0f)
            inputBuffer.putFloat(( pixel         and 0xFF) / 255.0f)
        }
        inputBuffer.rewind()
    }

    private fun runClassification(): Pair<Float, Float>? {
        val interp = interpreter ?: return null
        val outputShape = interp.getOutputTensor(0).shape()
        val numClasses = outputShape.getOrElse(1) { 2 }
        val output = Array(1) { FloatArray(numClasses) }

        val outputMap = HashMap<Int, Any>()
        outputMap[0] = output
        interp.runForMultipleInputsOutputs(arrayOf(inputBuffer), outputMap)

        val logits = output[0]
        // Numerically stable softmax
        val maxLogit = logits.max()
        val expVals  = logits.map { exp((it - maxLogit).toDouble()) }
        val expSum   = expVals.sum()
        val probs    = expVals.map { (it / expSum).toFloat() }

        return Pair(probs.getOrElse(0) { 0f }, probs.getOrElse(1) { 0f })
    }

    // -------------------------------------------------------------------------
    // Preprocessing helpers
    // -------------------------------------------------------------------------

    /** Gray World White Balance dengan strength blending terhadap original. */
    private fun grayWorldWB(src: Bitmap, strength: Float): Bitmap {
        val w = src.width; val h = src.height
        val pixels = IntArray(w * h)
        src.getPixels(pixels, 0, w, 0, 0, w, h)

        var sumR = 0L; var sumG = 0L; var sumB = 0L
        for (p in pixels) {
            sumR += (p shr 16) and 0xFF
            sumG += (p shr 8)  and 0xFF
            sumB += p          and 0xFF
        }
        val n = pixels.size.toFloat()
        val avgR = sumR / n; val avgG = sumG / n; val avgB = sumB / n
        val gray = (avgR + avgG + avgB) / 3.0

        val scaleR = if (avgR > 0) (gray / avgR).toFloat() else 1f
        val scaleG = if (avgG > 0) (gray / avgG).toFloat() else 1f
        val scaleB = if (avgB > 0) (gray / avgB).toFloat() else 1f

        val result = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        for (i in pixels.indices) {
            val p  = pixels[i]
            val r0 = (p shr 16) and 0xFF
            val g0 = (p shr 8)  and 0xFF
            val b0 =  p         and 0xFF
            // Blend antara corrected dan original sesuai strength
            val r = ((r0 * scaleR * strength + r0 * (1f - strength)).toInt()).coerceIn(0, 255)
            val g = ((g0 * scaleG * strength + g0 * (1f - strength)).toInt()).coerceIn(0, 255)
            val b = ((b0 * scaleB * strength + b0 * (1f - strength)).toInt()).coerceIn(0, 255)
            pixels[i] = Color.rgb(r, g, b)
        }
        result.setPixels(pixels, 0, w, 0, 0, w, h)
        return result
    }

    /**
     * Adaptive Gamma: hitung gamma berdasarkan rata-rata luminansi gambar.
     * Gambar gelap → gamma < 1 (perbriight), gambar terang → gamma > 1 (darken).
     * Gamma di-clamp ke [gammaMin, gammaMax].
     */
    private fun adaptiveGamma(src: Bitmap, gammaMin: Float, gammaMax: Float): Bitmap {
        val w = src.width; val h = src.height
        val pixels = IntArray(w * h)
        src.getPixels(pixels, 0, w, 0, 0, w, h)

        // Hitung rata-rata luminansi
        var sumLum = 0.0
        for (p in pixels) {
            val r = (p shr 16) and 0xFF
            val g = (p shr 8)  and 0xFF
            val b =  p         and 0xFF
            sumLum += 0.2126 * r + 0.7152 * g + 0.0722 * b
        }
        val avgLum = (sumLum / pixels.size / 255.0).coerceIn(0.01, 0.99)

        // Gamma inversely proportional ke luminansi
        // avgLum=0.5 (normal) → gamma=1.0; gelap → gamma<1; terang → gamma>1
        val gamma = (0.5 / avgLum).toFloat().coerceIn(gammaMin, gammaMax)

        // Build LUT
        val lut = IntArray(256) { i ->
            ((i / 255.0).pow(gamma.toDouble()) * 255.0).toInt().coerceIn(0, 255)
        }

        val result = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        for (i in pixels.indices) {
            val p = pixels[i]
            pixels[i] = Color.rgb(
                lut[(p shr 16) and 0xFF],
                lut[(p shr 8)  and 0xFF],
                lut[ p         and 0xFF]
            )
        }
        result.setPixels(pixels, 0, w, 0, 0, w, h)
        return result
    }

    /** Letterbox resize preserving aspect ratio, pad dengan (128,128,128). */
    private fun letterbox(src: Bitmap, targetSize: Int): Triple<Bitmap, Int, Int> {
        val origW = src.width; val origH = src.height
        val scale  = min(targetSize.toFloat() / origW, targetSize.toFloat() / origH)
        val scaledW = (origW * scale).toInt()
        val scaledH = (origH * scale).toInt()
        val padLeft = (targetSize - scaledW) / 2
        val padTop  = (targetSize - scaledH) / 2

        val scaled = Bitmap.createScaledBitmap(src, scaledW, scaledH, true)
        val result = Bitmap.createBitmap(targetSize, targetSize, Bitmap.Config.ARGB_8888)

        val canvas = android.graphics.Canvas(result)
        canvas.drawColor(Color.rgb(128, 128, 128))
        canvas.drawBitmap(scaled, padLeft.toFloat(), padTop.toFloat(), null)
        scaled.recycle()

        return Triple(result, padLeft, padTop)
    }

    /**
     * Bilateral filter approximation 3×3.
     * Menggunakan range weight berdasarkan intensitas — smoothing noise, preserve edge.
     * Ini bukan bilateral penuh (yang butuh O(r²) per pixel) tapi cukup untuk preprocessing.
     * Sigma_range ≈ 30 (dalam 0-255 space).
     */
    private fun bilateralFilterApprox(src: Bitmap): Bitmap {
        val w = src.width; val h = src.height
        val pixels = IntArray(w * h)
        src.getPixels(pixels, 0, w, 0, 0, w, h)
        val out = IntArray(w * h)

        val sigmaRange = 30f
        val sigmaRangeSq2 = 2f * sigmaRange * sigmaRange
        // Spatial weights 3×3 Gaussian, sigma=1
        val spatialW = floatArrayOf(
            0.0751f, 0.1238f, 0.0751f,
            0.1238f, 0.2042f, 0.1238f,
            0.0751f, 0.1238f, 0.0751f
        )

        for (y in 0 until h) {
            for (x in 0 until w) {
                val cp = pixels[y * w + x]
                val cR = (cp shr 16) and 0xFF
                val cG = (cp shr 8)  and 0xFF
                val cB =  cp         and 0xFF

                var sumR = 0f; var sumG = 0f; var sumB = 0f; var sumW = 0f

                for (ky in -1..1) {
                    for (kx in -1..1) {
                        val nx = (x + kx).coerceIn(0, w - 1)
                        val ny = (y + ky).coerceIn(0, h - 1)
                        val np = pixels[ny * w + nx]
                        val nR = (np shr 16) and 0xFF
                        val nG = (np shr 8)  and 0xFF
                        val nB =  np         and 0xFF

                        val dR = (nR - cR).toFloat()
                        val dG = (nG - cG).toFloat()
                        val dB = (nB - cB).toFloat()
                        val diffSq = dR * dR + dG * dG + dB * dB

                        val si = (ky + 1) * 3 + (kx + 1)
                        val w2 = spatialW[si] * exp(-diffSq / sigmaRangeSq2).toFloat()

                        sumR += nR * w2; sumG += nG * w2; sumB += nB * w2
                        sumW += w2
                    }
                }

                out[y * w + x] = Color.rgb(
                    (sumR / sumW).toInt().coerceIn(0, 255),
                    (sumG / sumW).toInt().coerceIn(0, 255),
                    (sumB / sumW).toInt().coerceIn(0, 255)
                )
            }
        }

        val result = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        result.setPixels(out, 0, w, 0, 0, w, h)
        return result
    }

    /**
     * Adaptive CLAHE di channel L* dari CIELAB.
     * Hanya L* yang di-enhance, a* dan b* (warna) dibiarkan agar warna konjungtiva tidak berubah.
     * Grid 8×8, clip limit adaptif berdasarkan rata-rata luminansi lokal.
     */
    private fun adaptiveCLAHE(src: Bitmap, clipMin: Float, clipMax: Float): Bitmap {
        val w = src.width; val h = src.height
        val pixels = IntArray(w * h)
        src.getPixels(pixels, 0, w, 0, 0, w, h)

        // Konversi ke CIELAB L* channel saja
        val lStar = FloatArray(w * h)
        for (i in pixels.indices) {
            val p = pixels[i]
            val r = (p shr 16) and 0xFF
            val g = (p shr 8)  and 0xFF
            val b =  p         and 0xFF
            // sRGB → linear
            val rL = srgbToLinear(r / 255f)
            val gL = srgbToLinear(g / 255f)
            val bL = srgbToLinear(b / 255f)
            // Linear RGB → Y (luminance)
            val Y = (0.2126f * rL + 0.7152f * gL + 0.0722f * bL)
                .coerceIn(0f, 1f)
            // Y → L*
            lStar[i] = labF(Y) * 116f - 16f  // L* range: 0–100
        }

        // Grid CLAHE 8×8
        val gridX = 8; val gridY = 8
        val tileW = w / gridX; val tileH = h / gridY

        // Build histogram & CDF per tile
        val histSize = 256
        for (gy in 0 until gridY) {
            for (gx in 0 until gridX) {
                val x0 = gx * tileW
                val y0 = gy * tileH
                val x1 = if (gx == gridX - 1) w else x0 + tileW
                val y1 = if (gy == gridY - 1) h else y0 + tileH
                val tilePixels = (x1 - x0) * (y1 - y0)

                // Hitung rata-rata luminansi lokal untuk clip adaptif
                var sumL = 0f
                for (ty in y0 until y1) {
                    for (tx in x0 until x1) {
                        sumL += lStar[ty * w + tx]
                    }
                }
                val avgL = sumL / tilePixels  // 0–100
                // Clip lebih tinggi di daerah gelap (more contrast needed)
                val clipLimit = (clipMax - (clipMax - clipMin) * (avgL / 100f)).coerceIn(clipMin, clipMax)

                // Histogram L* (bin 0–255 dari L* 0–100)
                val hist = IntArray(histSize)
                for (ty in y0 until y1) {
                    for (tx in x0 until x1) {
                        val bin = (lStar[ty * w + tx] / 100f * 255f).toInt().coerceIn(0, 255)
                        hist[bin]++
                    }
                }

                // Clip histogram
                val clipCount = (clipLimit * tilePixels / histSize).toInt().coerceAtLeast(1)
                var excess = 0
                for (b in 0 until histSize) {
                    if (hist[b] > clipCount) { excess += hist[b] - clipCount; hist[b] = clipCount }
                }
                val redistrib = excess / histSize
                val residual  = excess - redistrib * histSize
                for (b in 0 until histSize) hist[b] += redistrib
                for (b in 0 until residual)  hist[b]++

                // CDF → LUT
                val lut = FloatArray(histSize)
                var cdf = 0
                for (b in 0 until histSize) { cdf += hist[b]; lut[b] = cdf.toFloat() / tilePixels * 100f }

                // Apply LUT ke tile
                for (ty in y0 until y1) {
                    for (tx in x0 until x1) {
                        val bin = (lStar[ty * w + tx] / 100f * 255f).toInt().coerceIn(0, 255)
                        lStar[ty * w + tx] = lut[bin].coerceIn(0f, 100f)
                    }
                }
            }
        }

        // Konversi kembali: L* baru + a*b* original → RGB
        val result = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        for (i in pixels.indices) {
            val p = pixels[i]
            val rOrig = (p shr 16) and 0xFF
            val gOrig = (p shr 8)  and 0xFF
            val bOrig =  p         and 0xFF

            // Hitung a*, b* dari warna asli
            val rL = srgbToLinear(rOrig / 255f)
            val gL = srgbToLinear(gOrig / 255f)
            val bL = srgbToLinear(bOrig / 255f)
            val X = 0.4124f * rL + 0.3576f * gL + 0.1805f * bL
            val Y = 0.2126f * rL + 0.7152f * gL + 0.0722f * bL
            val Z = 0.0193f * rL + 0.1192f * gL + 0.9505f * bL

            val Xn = 0.95047f; val Yn = 1.00000f; val Zn = 1.08883f
            val aStar = 500f * (labF(X / Xn) - labF(Y / Yn))
            val bStar = 200f * (labF(Y / Yn) - labF(Z / Zn))

            // Reconstruct dengan L* baru
            val lNew = lStar[i]
            val fy = (lNew + 16f) / 116f
            val fx = aStar / 500f + fy
            val fz = fy - bStar / 200f
            val Xr = if (fx > 0.2069f) fx * fx * fx else (fx - 16f / 116f) / 7.787f
            val Yr = if (fy > 0.2069f) fy * fy * fy else (fy - 16f / 116f) / 7.787f
            val Zr = if (fz > 0.2069f) fz * fz * fz else (fz - 16f / 116f) / 7.787f

            val xN = Xr * Xn; val yN = Yr * Yn; val zN = Zr * Zn

            // XYZ → linear RGB
            val rLin = ( 3.2406f * xN - 1.5372f * yN - 0.4986f * zN).coerceIn(0f, 1f)
            val gLin = (-0.9689f * xN + 1.8758f * yN + 0.0415f * zN).coerceIn(0f, 1f)
            val bLin = ( 0.0557f * xN - 0.2040f * yN + 1.0570f * zN).coerceIn(0f, 1f)

            pixels[i] = Color.rgb(
                (linearToSrgb(rLin) * 255f).toInt().coerceIn(0, 255),
                (linearToSrgb(gLin) * 255f).toInt().coerceIn(0, 255),
                (linearToSrgb(bLin) * 255f).toInt().coerceIn(0, 255)
            )
        }
        result.setPixels(pixels, 0, w, 0, 0, w, h)
        return result
    }

    // CIE Lab helper functions
    private fun labF(t: Float): Float {
        val delta = 6f / 29f
        return if (t > delta * delta * delta) t.pow(1f / 3f)
        else t / (3f * delta * delta) + 4f / 29f
    }

    private fun srgbToLinear(c: Float): Float =
        if (c <= 0.04045f) c / 12.92f else ((c + 0.055f) / 1.055f).pow(2.4f)

    private fun linearToSrgb(c: Float): Float =
        if (c <= 0.0031308f) c * 12.92f else 1.055f * c.pow(1f / 2.4f) - 0.055f

    private fun loadModelFile(context: Context, modelPath: String): MappedByteBuffer {
        val assetFd = context.assets.openFd(modelPath)
        return FileInputStream(assetFd.fileDescriptor).channel.map(
            FileChannel.MapMode.READ_ONLY,
            assetFd.startOffset,
            assetFd.declaredLength
        )
    }

    fun close() {
        engine.close()
        interpreter = null
    }
}
