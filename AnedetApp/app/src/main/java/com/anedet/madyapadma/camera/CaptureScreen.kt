package com.anedet.madyapadma.camera

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Matrix
import android.util.Log
import android.widget.Toast
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.Camera
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.CameraAlt
import androidx.compose.material.icons.filled.FlashOff
import androidx.compose.material.icons.filled.FlashOn
import androidx.compose.material.icons.filled.Menu
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.LocalLifecycleOwner as ComposeLifecycleOwner
import androidx.lifecycle.lifecycleScope
import androidx.lifecycle.viewmodel.compose.viewModel
import com.anedet.madyapadma.model.MaskData
import com.anedet.madyapadma.ui.components.t
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.collectLatest
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicReference

@Composable
fun CaptureScreen(
    onResult: (String) -> Unit,
    onSettings: () -> Unit,
    onOpenDrawer: () -> Unit,
    viewModel: CameraViewModel = viewModel()
) {
    val context = LocalContext.current
    val lifecycleOwner = ComposeLifecycleOwner.current
    val isCapturing = remember { AtomicBoolean(false) }
    var torchEnabled by remember { mutableStateOf(false) }
    var cameraRef by remember { mutableStateOf<Camera?>(null) }
    var hasCameraPermission by remember {
        mutableStateOf(
            ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA) ==
                PackageManager.PERMISSION_GRANTED
        )
    }

    val permissionLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.RequestPermission()
    ) { granted ->
        hasCameraPermission = granted
        if (!granted) {
            Toast.makeText(context, "Camera permission required", Toast.LENGTH_LONG).show()
        }
    }

    val mainExecutor = remember { ContextCompat.getMainExecutor(context) }
    val analyzerExecutor = remember { Executors.newSingleThreadExecutor() }
    val imageCaptureRef = remember { AtomicReference<ImageCapture?>() }
    var overlayView by remember { mutableStateOf<MaskOverlayView?>(null) }
    
    // Bitmap pool untuk rotation reuse (Bug 5.2 fix)
    val bitmapPool = remember { com.anedet.madyapadma.ml.BitmapPool(maxPoolSize = 3) }

    val segmentor = remember(viewModel) { viewModel.segmentor }
    val autoCaptureStatus by viewModel.autoCaptureStatus.collectAsState()
    val autoCaptureProgress by viewModel.autoCaptureProgress.collectAsState()
    val autoCaptureEnabled by viewModel.settings.smartAutoCapture.collectAsState()

    val stabilityState = remember { StabilityTracker() }
    
    // Mask persistence - keep showing mask even if temporarily lost
    var lastValidMask by remember { mutableStateOf<MaskData?>(null) }
    var lastValidMaskBitmap by remember { mutableStateOf<Bitmap?>(null) }
    var maskLostFrames by remember { mutableStateOf(0) }
    val MASK_PERSISTENCE_FRAMES = 5  // Keep mask for 5 frames after lost
    
    // Cleanup bitmap pool on disposal
    androidx.compose.runtime.DisposableEffect(Unit) {
        onDispose {
            bitmapPool.clear()
            analyzerExecutor.shutdown()
        }
    }

    LaunchedEffect(Unit) {
        withContext(Dispatchers.Default) { segmentor.initialize() }
    }

    val captureDir = remember { context.cacheDir.absolutePath }

    LaunchedEffect(Unit) {
        viewModel.autoCaptureRequests.collectLatest { _ ->
            if (isCapturing.get()) return@collectLatest
            val ic = imageCaptureRef.get() ?: return@collectLatest
            isCapturing.set(true)
            viewModel.reportAutoCaptureStatus("capturing")
            val file = File(context.cacheDir, "capture_${System.currentTimeMillis()}.jpg")
            val outputOptions = ImageCapture.OutputFileOptions.Builder(file).build()
            ic.takePicture(
                outputOptions,
                mainExecutor,
                object : ImageCapture.OnImageSavedCallback {
                    override fun onImageSaved(output: ImageCapture.OutputFileResults) {
                        isCapturing.set(false)
                        stabilityState.reset()
                        onResult(file.absolutePath)
                    }
                    override fun onError(exception: ImageCaptureException) {
                        isCapturing.set(false)
                        Toast.makeText(context, "Capture failed: ${exception.message}", Toast.LENGTH_SHORT).show()
                    }
                }
            )
        }
    }

    if (!hasCameraPermission) {
        Column(
            modifier = Modifier.fillMaxSize(),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.Center
        ) {
            Text("Camera permission is needed", style = MaterialTheme.typography.bodyLarge)
            Button(
                onClick = { permissionLauncher.launch(Manifest.permission.CAMERA) },
                modifier = Modifier.padding(top = 16.dp)
            ) { Text("Grant Permission") }
        }
    } else {
        Box(modifier = Modifier.fillMaxSize()) {
            AndroidView(
                factory = { ctx ->
                    val previewView = PreviewView(ctx).apply {
                        scaleType = PreviewView.ScaleType.FILL_CENTER
                    }
                    val overlay = MaskOverlayView(ctx)
                    overlayView = overlay

                    val cameraProviderFuture = ProcessCameraProvider.getInstance(ctx)
                    cameraProviderFuture.addListener({
                        val cameraProvider = cameraProviderFuture.get()
                        val preview = Preview.Builder().build().also {
                            it.setSurfaceProvider(previewView.surfaceProvider)
                        }

                        val imageCapture = ImageCapture.Builder()
                            .setCaptureMode(ImageCapture.CAPTURE_MODE_MAXIMIZE_QUALITY)
                            .build()
                        imageCaptureRef.set(imageCapture)

                        // Bug 5.1 Fix: Set explicit resolution untuk avoid 4K overhead
                        // Low-end device target: 1280×720 optimal balance quality vs performance
                        val imageAnalysis = ImageAnalysis.Builder()
                            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                            .setTargetRotation(previewView.display?.rotation ?: android.view.Surface.ROTATION_0)
                            .setTargetResolution(android.util.Size(1280, 720))
                            .build()

                        val isProcessing = AtomicBoolean(false)

                        imageAnalysis.setAnalyzer(analyzerExecutor) { imageProxy ->
                            // Bug 2.3 Fix: Proper non-blocking dengan early return jika masih processing
                            if (isProcessing.get()) {
                                imageProxy.close()
                                return@setAnalyzer
                            }
                            
                            isProcessing.set(true)
                            
                            val rotation = imageProxy.imageInfo.rotationDegrees
                            var srcBitmap: Bitmap? = null
                            var bitmap: Bitmap? = null
                            var maskBmp: Bitmap? = null
                            
                            try {
                                // Bug 1.1 Fix: Proper try-finally untuk bitmap recycle
                                srcBitmap = try {
                                    imageProxy.toBitmap()
                                } catch (e: Exception) {
                                    Log.e(TAG, "toBitmap failed: ${e.message}")
                                    null
                                }
                                
                                if (srcBitmap == null) {
                                    return@setAnalyzer
                                }
                                
                                if (!segmentor.isReady()) {
                                    kotlinx.coroutines.runBlocking { segmentor.initialize() }
                                }

                                // Bug 5.2 Fix: Use bitmap pool for rotation
                                bitmap = if (rotation != 0) {
                                    val matrix = Matrix().apply { postRotate(rotation.toFloat()) }
                                    
                                    // Try get from pool first
                                    val pooled = bitmapPool.get(
                                        srcBitmap.height, // rotated dimensions
                                        srcBitmap.width,
                                        Bitmap.Config.ARGB_8888
                                    )
                                    
                                    val rotated = if (pooled != null) {
                                        // Reuse pooled bitmap
                                        val canvas = android.graphics.Canvas(pooled)
                                        canvas.drawBitmap(srcBitmap, matrix, null)
                                        pooled
                                    } else {
                                        // Create new if pool empty
                                        Bitmap.createBitmap(
                                            srcBitmap, 0, 0,
                                            srcBitmap.width, srcBitmap.height, matrix, true
                                        )
                                    }
                                    
                                    rotated
                                } else {
                                    srcBitmap
                                }
                                
                                // Bug 2.4 Fix: Quality pre-check sebelum inference
                                val minSharpness = viewModel.settings.sharpnessMin.value
                                if (!ImageQualityUtils.isQualitySufficientForInference(bitmap, minSharpness)) {
                                    // Skip inference on low-quality frame
                                    mainExecutor.execute {
                                        overlayView?.setMaskData(null, null, bitmap.width, bitmap.height, 0)
                                        if (viewModel.settings.smartAutoCapture.value && !isCapturing.get()) {
                                            viewModel.reportAutoCaptureStatus("low_quality", 0)
                                        }
                                    }
                                    return@setAnalyzer
                                }

                                val result: MaskData? = segmentor.runSegmentation(bitmap)

                                // 4. Create mask bitmap on background thread
                                if (result != null && result.confidence >= viewModel.settings.confidenceThreshold.value) {
                                    val mw = result.protoW; val mh = result.protoH
                                    if (mw > 0 && mh > 0) {
                                        val bmp = Bitmap.createBitmap(mw, mh, Bitmap.Config.ARGB_8888)
                                        val pixels = IntArray(mw * mh)
                                        for (y in 0 until mh) {
                                            for (x in 0 until mw) {
                                                val alpha = if (result.mask[y][x] > 0.5f) 160 else 0
                                                pixels[y * mw + x] = android.graphics.Color.argb(alpha, 76, 175, 80)
                                            }
                                        }
                                        bmp.setPixels(pixels, 0, mw, 0, 0, mw, mh)
                                        maskBmp = bmp
                                    }
                                }

                                // Post UI updates to Main Thread
                                mainExecutor.execute {
                                    val confThreshold = viewModel.settings.confidenceThreshold.value
                                    val finalResult = if (result != null && result.confidence >= confThreshold) result else null
                                    
                                    // Mask persistence logic - reduce flicker
                                    val displayResult: MaskData?
                                    val displayMaskBmp: Bitmap?
                                    
                                    if (finalResult != null) {
                                        // New valid mask detected
                                        lastValidMask = finalResult
                                        lastValidMaskBitmap?.recycle()  // Recycle old
                                        lastValidMaskBitmap = maskBmp
                                        maskLostFrames = 0
                                        displayResult = finalResult
                                        displayMaskBmp = maskBmp
                                    } else {
                                        // No mask detected - check if we should persist
                                        maskLostFrames++
                                        if (maskLostFrames <= MASK_PERSISTENCE_FRAMES && lastValidMask != null) {
                                            // Still within persistence window - show last valid mask
                                            displayResult = lastValidMask
                                            displayMaskBmp = lastValidMaskBitmap
                                        } else {
                                            // Too long without detection - clear mask
                                            lastValidMask = null
                                            lastValidMaskBitmap?.recycle()
                                            lastValidMaskBitmap = null
                                            maskLostFrames = 0
                                            displayResult = null
                                            displayMaskBmp = null
                                        }
                                    }
                                    
                                    overlayView?.setMaskData(displayResult, displayMaskBmp, bitmap.width, bitmap.height, 0)
                                    viewModel.updateLiveMask(displayResult, bitmap.width, bitmap.height)

                                    val autoCap = viewModel.settings.smartAutoCapture.value
                                    if (autoCap && result != null && !isCapturing.get()) {
                                        val okConf = result.confidence >= confThreshold
                                        val sharpness = ImageQualityUtils.calculateBlurriness(bitmap)
                                        val sharpOk = sharpness >= viewModel.settings.sharpnessMin.value
                                        val sizeOk = (result.bbox.width() * result.bbox.height()) >=
                                                (bitmap.width * bitmap.height) * 0.01f
                                        
                                        val stable = stabilityState.update(
                                            detected = okConf && sharpOk && sizeOk,
                                            stabilityNeeded = viewModel.settings.stabilityFrames.value
                                        )

                                        when {
                                            !okConf -> {
                                                stabilityState.reset()
                                                viewModel.reportAutoCaptureStatus("searching", 0)
                                            }
                                            !sharpOk -> {
                                                stabilityState.reset()
                                                viewModel.reportAutoCaptureStatus("low_quality", 0)
                                            }
                                            !sizeOk -> {
                                                stabilityState.reset()
                                                viewModel.reportAutoCaptureStatus("low_quality", 0)
                                            }
                                            stable -> {
                                                viewModel.reportAutoCaptureStatus("ready", 100)
                                                viewModel.requestAutoCapture(captureDir)
                                                stabilityState.reset()
                                            }
                                            else -> {
                                                viewModel.reportAutoCaptureStatus(
                                                    "stabilizing",
                                                    stabilityState.count * 100 / viewModel.settings.stabilityFrames.value
                                                )
                                            }
                                        }
                                    } else if (!autoCap) {
                                        viewModel.reportAutoCaptureStatus("searching", 0)
                                    }
                                }
                            } catch (e: Exception) {
                                Log.e(TAG, "Analyzer error: ${e.message}")
                            } finally {
                                imageProxy.close()
                                
                                // Recycle atau return to pool
                                if (bitmap != null && bitmap !== srcBitmap && rotation != 0) {
                                    // Return rotated bitmap to pool
                                    bitmapPool.put(bitmap)
                                } else {
                                    bitmap?.recycle()
                                }
                                
                                srcBitmap?.recycle()
                                
                                isProcessing.set(false)
                            }
                        }

                        cameraProvider.unbindAll()
                        val camera = cameraProvider.bindToLifecycle(
                            lifecycleOwner,
                            CameraSelector.DEFAULT_BACK_CAMERA,
                            preview,
                            imageCapture,
                            imageAnalysis
                        )
                        cameraRef = camera
                    }, mainExecutor)

                    previewView
                },
                modifier = Modifier.fillMaxSize()
            )

            overlayView?.let { overlay ->
                AndroidView(factory = { overlay }, modifier = Modifier.fillMaxSize())
            }

            // UI Buttons (Drawer, Settings, Flash)
            IconButton(
                onClick = onOpenDrawer,
                modifier = Modifier
                    .align(Alignment.TopStart)
                    .padding(top = 48.dp, start = 16.dp)
                    .size(48.dp)
                    .background(Color.Black.copy(alpha = 0.4f), CircleShape)
                    .clip(CircleShape)
            ) {
                Icon(Icons.Default.Menu, contentDescription = "Menu", tint = Color.White)
            }

            IconButton(
                onClick = onSettings,
                modifier = Modifier
                    .align(Alignment.TopStart)
                    .padding(top = 48.dp, start = 80.dp)
                    .size(48.dp)
                    .background(Color.Black.copy(alpha = 0.4f), CircleShape)
                    .clip(CircleShape)
            ) {
                Icon(Icons.Default.Settings, contentDescription = "Settings", tint = Color.White)
            }

            IconButton(
                onClick = {
                    torchEnabled = !torchEnabled
                    cameraRef?.cameraControl?.enableTorch(torchEnabled)
                },
                modifier = Modifier
                    .align(Alignment.TopEnd)
                    .padding(top = 48.dp, end = 16.dp)
                    .size(48.dp)
                    .background(Color.Black.copy(alpha = 0.4f), CircleShape)
                    .clip(CircleShape)
            ) {
                Icon(
                    imageVector = if (torchEnabled) Icons.Default.FlashOn else Icons.Default.FlashOff,
                    contentDescription = "Flash",
                    tint = if (torchEnabled) Color(0xFFFFD600) else Color.White
                )
            }

            AnimatedVisibility(
                visible = autoCaptureEnabled,
                enter = fadeIn(), exit = fadeOut(),
                modifier = Modifier.align(Alignment.TopCenter).padding(top = 110.dp)
            ) {
                AutoCaptureBanner(status = autoCaptureStatus, progress = autoCaptureProgress)
            }

            IconButton(
                onClick = {
                    if (isCapturing.get()) return@IconButton
                    val ic = imageCaptureRef.get() ?: return@IconButton
                    isCapturing.set(true)
                    viewModel.reportAutoCaptureStatus("capturing")
                    val file = File(context.cacheDir, "capture_${System.currentTimeMillis()}.jpg")
                    val outputOptions = ImageCapture.OutputFileOptions.Builder(file).build()
                    ic.takePicture(
                        outputOptions,
                        mainExecutor,
                        object : ImageCapture.OnImageSavedCallback {
                            override fun onImageSaved(output: ImageCapture.OutputFileResults) {
                                isCapturing.set(false); stabilityState.reset(); onResult(file.absolutePath)
                            }
                            override fun onError(exception: ImageCaptureException) {
                                isCapturing.set(false)
                                Toast.makeText(context, "Capture failed: ${exception.message}", Toast.LENGTH_SHORT).show()
                            }
                        }
                    )
                },
                modifier = Modifier
                    .align(Alignment.BottomCenter)
                    .padding(bottom = 48.dp)
                    .size(72.dp)
                    .background(Color(0xFF9C27B0), RoundedCornerShape(20.dp))
                    .clip(RoundedCornerShape(20.dp)),
                enabled = !isCapturing.get()
            ) {
                Icon(Icons.Default.CameraAlt, contentDescription = "Capture", tint = Color.White, modifier = Modifier.size(36.dp))
            }

            if (isCapturing.get()) {
                Box(Modifier.fillMaxSize().background(Color.Black.copy(alpha = 0.3f)), Alignment.Center) {
                    CircularProgressIndicator(color = MaterialTheme.colorScheme.primary)
                }
            }
        }
    }
}

@Composable
private fun AutoCaptureBanner(
    status: String,
    progress: Int
) {
    val (color, text) = when (status) {
        "searching" -> Color(0xFF607D8B) to t("auto_status_searching")
        "stabilizing" -> Color(0xFFFFA000) to t("auto_status_stabilizing")
        "low_quality" -> Color(0xFFE53935) to t("auto_status_low_quality")
        "ready" -> Color(0xFF43A047) to t("auto_status_ready")
        "capturing" -> Color(0xFF6750A4) to t("auto_status_capturing")
        else -> Color(0xFF607D8B) to status
    }

    Surface(
        color = color.copy(alpha = 0.85f),
        shape = RoundedCornerShape(24.dp),
        shadowElevation = 4.dp
    ) {
        Column(
            modifier = Modifier
                .padding(horizontal = 20.dp, vertical = 12.dp)
                .width(220.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Text(
                text = text,
                color = Color.White,
                fontWeight = FontWeight.SemiBold,
                style = MaterialTheme.typography.bodyMedium
            )
            if (status == "stabilizing" || status == "ready") {
                Spacer(modifier = Modifier.height(6.dp))
                LinearProgressIndicator(
                    progress = { (progress / 100f).coerceIn(0f, 1f) },
                    modifier = Modifier.fillMaxWidth().height(4.dp),
                    color = Color.White,
                    trackColor = Color.White.copy(alpha = 0.3f)
                )
            }
        }
    }
}

/**
 * Simple Stability Tracker untuk auto-capture.
 * 
 * Tracks detection success dalam sliding window untuk memastikan
 * frame stabil sebelum trigger auto-capture.
 */
private class StabilityTracker {
    private val windowSize = 8
    private val window = ArrayDeque<Boolean>(windowSize)
    var count: Int = 0
        private set

    fun update(detected: Boolean, stabilityNeeded: Int): Boolean {
        if (window.size == windowSize) window.removeFirst()
        window.addLast(detected)
        count = window.count { it }
        return count >= stabilityNeeded
    }

    fun reset() {
        window.clear()
        count = 0
    }
}

private const val TAG = "CaptureScreen"
