package com.anedet.madyapadma.native.jni

object NativeBridge {
    init {
        System.loadLibrary("anedet_native")
    }

    external fun runSegmentation(imageBuffer: ByteArray, width: Int, height: Int): FloatArray
    external fun postProcessMask(rawMask: FloatArray, imgW: Int, imgH: Int): FloatArray
    external fun cropAndPreprocess(
        sourceImage: ByteArray,
        left: Float, top: Float,
        right: Float, bottom: Float,
        targetW: Int, targetH: Int
    ): FloatArray

    external fun runClassification(inputBuffer: FloatArray): FloatArray
}
