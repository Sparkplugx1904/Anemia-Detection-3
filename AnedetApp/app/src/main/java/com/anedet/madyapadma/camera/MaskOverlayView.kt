package com.anedet.madyapadma.camera

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.RectF
import android.util.AttributeSet
import android.util.Log
import android.view.View
import com.anedet.madyapadma.model.MaskData
import com.anedet.madyapadma.ml.Segmentor
import kotlin.math.max

class MaskOverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {

    private var maskData: MaskData? = null
    private var imgW: Int = 0
    private var imgH: Int = 0
    private var rotation: Int = 0
    private var maskBitmap: Bitmap? = null
    private var polygonPath: android.graphics.Path? = null

    private val maskPaint = Paint().apply {
        isFilterBitmap = false // Paksa nearest-neighbor agar tajam
        alpha = 130
    }
    private val polygonPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.argb(255, 0, 230, 118) // Hijau neon
        style = Paint.Style.STROKE
        strokeWidth = 5f
    }
    private val bboxPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.YELLOW
        style = Paint.Style.STROKE
        strokeWidth = 3f
        pathEffect = android.graphics.DashPathEffect(floatArrayOf(10f, 10f), 0f)
    }
    private val labelPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        textSize = 36f
        isFakeBoldText = true
        setShadowLayer(4f, 1f, 1f, Color.BLACK)
    }

    fun setMaskData(
        data: MaskData?,
        maskBitmap: Bitmap?,
        imageWidth: Int = imgW,
        imageHeight: Int = imgH,
        imageRotation: Int = rotation
    ) {
        this.maskData = data
        this.imgW = imageWidth
        this.imgH = imageHeight
        this.rotation = imageRotation
        
        this.maskBitmap?.recycle()
        this.maskBitmap = maskBitmap

        polygonPath = data?.let { extractPolygonPath(it) }
        invalidate()
    }

    private fun extractPolygonPath(data: MaskData): android.graphics.Path {
        val polyPath = android.graphics.Path()
        val mask = data.mask
        val h = mask.size
        val w = if (h > 0) mask[0].size else 0
        if (w == 0) return polyPath

        var started = false
        // Sisi kiri
        for (y in 0 until h step 2) {
            for (x in 0 until w) {
                if (mask[y][x] > 0.5f) {
                    if (!started) { polyPath.moveTo(x.toFloat(), y.toFloat()); started = true }
                    else polyPath.lineTo(x.toFloat(), y.toFloat())
                    break
                }
            }
        }
        // Sisi kanan
        for (y in h - 1 downTo 0 step 2) {
            for (x in w - 1 downTo 0) {
                if (mask[y][x] > 0.5f) {
                    polyPath.lineTo(x.toFloat(), y.toFloat())
                    break
                }
            }
        }
        if (started) polyPath.close()
        return polyPath
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val data = maskData ?: return
        if (imgW <= 0 || imgH <= 0 || width == 0 || height == 0) return

        // --- MATRIKS TUNGGAL: UNIFIED COORDINATE MAPPING ---
        val viewMatrix = Matrix()

        // 1. Skala Preview (FILL_CENTER)
        val rotW: Float; val rotH: Float
        if (rotation == 90 || rotation == 270) {
            rotW = imgH.toFloat(); rotH = imgW.toFloat()
        } else {
            rotW = imgW.toFloat(); rotH = imgH.toFloat()
        }
        val scale = max(width.toFloat() / rotW, height.toFloat() / rotH)
        val offX = (width.toFloat() - rotW * scale) / 2f
        val offY = (height.toFloat() - rotH * scale) / 2f

        viewMatrix.postTranslate(offX, offY)
        viewMatrix.preScale(scale, scale)

        // 2. Rotasi Gambar
        if (rotation != 0) {
            viewMatrix.preTranslate(rotW / 2f, rotH / 2f)
            viewMatrix.preRotate(-rotation.toFloat())
            viewMatrix.preTranslate(-rotW / 2f, -rotH / 2f)
        }

        // 3. Model Space -> Original Image Space
        val modelToOrig = Matrix()
        modelToOrig.postScale(1f / data.lbScale, 1f / data.lbScale)
        modelToOrig.postTranslate(-data.lbPadLeft / data.lbScale, -data.lbPadTop / data.lbScale)
        viewMatrix.preConcat(modelToOrig)

        // 4. Proto Space -> Model Space
        val protoToModel = Matrix()
        val mw = data.protoW.coerceAtLeast(1)
        val mh = data.protoH.coerceAtLeast(1)
        protoToModel.postScale(Segmentor.INPUT_SIZE.toFloat() / mw, Segmentor.INPUT_SIZE.toFloat() / mh)

        val finalMaskMatrix = Matrix(viewMatrix)
        finalMaskMatrix.preConcat(protoToModel)

        // --- DRAWING ---
        // A. Mask Area (Solid Green)
        maskBitmap?.let { bmp ->
            canvas.drawBitmap(bmp, finalMaskMatrix, maskPaint)
        }

        // B. Polygon Line (Neon Vector)
        polygonPath?.let { rawPath ->
            val screenPath = android.graphics.Path()
            screenPath.addPath(rawPath, finalMaskMatrix)
            canvas.drawPath(screenPath, polygonPaint)
        }

        // C. Bounding Box (Dashed Yellow)
        val screenBbox = RectF()
        viewMatrix.mapRect(screenBbox, data.bbox)
        canvas.drawRect(screenBbox, bboxPaint)
        canvas.drawText("${(data.confidence * 100).toInt()}%", screenBbox.left, screenBbox.top - 10f, labelPaint)
    }

    companion object {
        private const val TAG = "MaskOverlay"
    }
}