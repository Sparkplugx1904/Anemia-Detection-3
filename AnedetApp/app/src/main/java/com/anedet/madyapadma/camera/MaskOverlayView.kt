package com.anedet.madyapadma.camera

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.AttributeSet
import android.util.Log
import android.view.View
import com.anedet.madyapadma.model.MaskData
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

    private val fillPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.argb(140, 76, 175, 80)
        style = Paint.Style.FILL
    }
    private val bboxPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.YELLOW
        style = Paint.Style.STROKE
        strokeWidth = 8f
    }
    private val cornerPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.argb(255, 255, 193, 7)
        style = Paint.Style.STROKE
        strokeWidth = 14f
    }
    private val labelPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        textSize = 38f
        isFakeBoldText = true
        setShadowLayer(4f, 1f, 1f, Color.BLACK)
    }

    fun setMaskData(
        data: MaskData?,
        imageWidth: Int = imgW,
        imageHeight: Int = imgH,
        imageRotation: Int = rotation
    ) {
        this.maskData = data
        this.imgW = imageWidth
        this.imgH = imageHeight
        this.rotation = imageRotation
        Log.d(TAG, "setMaskData: data=${data != null} conf=${data?.confidence} bbox=${data?.bbox} img=${imgW}x$imgH rot=$rotation")
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val data = maskData ?: return
        if (imgW <= 0 || imgH <= 0 || width == 0 || height == 0) return

        canvas.save()

        // Dimensi setelah rotasi diterapkan
        val rotW: Float
        val rotH: Float
        if (rotation == 90 || rotation == 270) {
            rotW = imgH.toFloat()
            rotH = imgW.toFloat()
        } else {
            rotW = imgW.toFloat()
            rotH = imgH.toFloat()
        }

        // FILL_CENTER: scale untuk memenuhi view, pertahankan aspect ratio
        val viewW = width.toFloat()
        val viewH = height.toFloat()
        val scale = max(viewW / rotW, viewH / rotH)
        val dispW = rotW * scale
        val dispH = rotH * scale
        val offX = (viewW - dispW) / 2f
        val offY = (viewH - dispH) / 2f

        canvas.translate(offX, offY)
        canvas.scale(scale, scale)

        // Un-rotate agar kembali ke koordinat gambar asli
        if (rotation != 0) {
            canvas.translate(rotW / 2f, rotH / 2f)
            canvas.rotate(-rotation.toFloat())
            canvas.translate(-rotW / 2f, -rotH / 2f)
        }

        // Gambar ellipse fill + bbox + corner accents
        val b = data.bbox
        if (b.width() > 0 && b.height() > 0) {
            // Fill area (semi-transparan)
            val cx = b.centerX()
            val cy = b.centerY()
            val rx = b.width() / 2f
            val ry = b.height() / 2f
            canvas.drawOval(cx - rx, cy - ry, cx + rx, cy + ry, fillPaint)

            // Bbox outline (putus-putus simulasi)
            canvas.drawRect(b, bboxPaint)

            // Corner accents
            val cornerLen = minOf(rx, ry) * 0.35f
            val l = b.left; val t = b.top; val r = b.right; val bt = b.bottom
            // top-left
            canvas.drawLine(l, t, l + cornerLen, t, cornerPaint)
            canvas.drawLine(l, t, l, t + cornerLen, cornerPaint)
            // top-right
            canvas.drawLine(r, t, r - cornerLen, t, cornerPaint)
            canvas.drawLine(r, t, r, t + cornerLen, cornerPaint)
            // bottom-left
            canvas.drawLine(l, bt, l + cornerLen, bt, cornerPaint)
            canvas.drawLine(l, bt, l, bt - cornerLen, cornerPaint)
            // bottom-right
            canvas.drawLine(r, bt, r - cornerLen, bt, cornerPaint)
            canvas.drawLine(r, bt, r, bt - cornerLen, cornerPaint)

            // Label
            val confText = "${(data.confidence * 100).toInt()}%"
            canvas.drawText(confText, l, t - 10f, labelPaint)
        }

        canvas.restore()
    }

    companion object {
        private const val TAG = "MaskOverlay"
    }
}
