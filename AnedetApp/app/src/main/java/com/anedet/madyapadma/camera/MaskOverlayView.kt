package com.anedet.madyapadma.camera

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.PorterDuff
import android.graphics.PorterDuffXfermode
import android.util.AttributeSet
import android.view.View
import com.anedet.madyapadma.model.MaskData

class MaskOverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {

    private var maskData: MaskData? = null
    private val paint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.GREEN
        alpha = 100
        style = Paint.Style.FILL
    }
    private val bboxPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.YELLOW
        style = Paint.Style.STROKE
        strokeWidth = 5f
    }

    fun setMaskData(data: MaskData?) {
        this.maskData = data
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val data = maskData ?: return

        // Scale bbox to view size
        val scaleX = width.toFloat() / 320f // Assuming model input 320
        val scaleY = height.toFloat() / 320f

        // This is a simplified visualization for live preview
        // In real app, we should use the same matrix logic as AnemiaPipeline
        val left = data.bbox.left * scaleX
        val top = data.bbox.top * scaleY
        val right = data.bbox.right * scaleX
        val bottom = data.bbox.bottom * scaleY

        canvas.drawRect(left, top, right, bottom, bboxPaint)
        canvas.drawRect(left, top, right, bottom, paint)
    }
}
