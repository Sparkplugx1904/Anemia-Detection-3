package com.anedet.madyapadma.camera

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.Path
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

    private val maskPaint = Paint().apply {
        isFilterBitmap = false
        alpha = 130
    }
    private val polygonPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.YELLOW
        style = Paint.Style.STROKE
        strokeWidth = 5f
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

        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val data = maskData ?: return
        if (imgW <= 0 || imgH <= 0 || width == 0 || height == 0) return

        // --- UNIFIED COORDINATE MAPPING ---
        val viewMatrix = Matrix()

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

        if (rotation != 0) {
            viewMatrix.preTranslate(rotW / 2f, rotH / 2f)
            viewMatrix.preRotate(-rotation.toFloat())
            viewMatrix.preTranslate(-rotW / 2f, -rotH / 2f)
        }

        // --- DRAWING ---
        // A. Mask fill (proto space -> original space -> screen)
        val protoToModel = Matrix()
        val mw = data.protoW.coerceAtLeast(1)
        val mh = data.protoH.coerceAtLeast(1)
        protoToModel.postScale(Segmentor.INPUT_SIZE.toFloat() / mw, Segmentor.INPUT_SIZE.toFloat() / mh)
        val modelToOrig = Matrix()
        modelToOrig.postScale(1f / data.lbScale, 1f / data.lbScale)
        modelToOrig.postTranslate(-data.lbPadLeft / data.lbScale, -data.lbPadTop / data.lbScale)

        val maskMatrix = Matrix(viewMatrix)
        maskMatrix.preConcat(modelToOrig)
        maskMatrix.preConcat(protoToModel)

        maskBitmap?.let { bmp ->
            canvas.drawBitmap(bmp, maskMatrix, maskPaint)
        }

        // B. Polygon outline — titik sudah dalam koordinat original
        if (data.polygon.size >= 3) {
            val polyPath = polygonToPath(data.polygon, viewMatrix)
            canvas.drawPath(polyPath, polygonPaint)

            // Label confidence di atas titik polygon pertama
            val first = data.polygon.first()
            val labelPt = floatArrayOf(first.x, first.y)
            viewMatrix.mapPoints(labelPt)
            canvas.drawText(
                "${(data.confidence * 100).toInt()}%",
                labelPt[0],
                (labelPt[1] - 10f).coerceAtLeast(40f),
                labelPaint
            )
        }
    }

    private fun polygonToPath(points: List<android.graphics.PointF>, map: Matrix): Path {
        val path = Path()
        if (points.isEmpty()) return path

        val coords = FloatArray(points.size * 2)
        for (i in points.indices) {
            coords[i * 2] = points[i].x
            coords[i * 2 + 1] = points[i].y
        }
        map.mapPoints(coords)

        path.moveTo(coords[0], coords[1])
        var i = 2
        while (i < coords.size) {
            path.lineTo(coords[i], coords[i + 1])
            i += 2
        }
        path.close()
        return path
    }

    companion object {
        private const val TAG = "MaskOverlay"
    }
}
