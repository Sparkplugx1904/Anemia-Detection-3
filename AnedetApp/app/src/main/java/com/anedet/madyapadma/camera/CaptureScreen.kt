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
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.LocalLifecycleOwner as ComposeLifecycleOwner
import androidx.lifecycle.lifecycleScope
import androidx.lifecycle.viewmodel.compose.viewModel
import com.anedet.madyapadma.R
import com.anedet.madyapadma.model.MaskData
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

    val segmentor = remember(viewModel) { viewModel.segmentor }
    val autoCaptureStatus by viewModel.autoCaptureStatus.collectAsState()
    val autoCaptureProgress by viewModel.autoCaptureProgress.collectAsState()
    val autoCaptureEnabled by viewModel.settings.smartAutoCapture.collectAsState()

    val stabilityState = remember { StabilityTracker() }

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

                        val imageAnalysis = ImageAnalysis.Builder()
                            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                            .setTargetRotation(previewView.display?.rotation ?: android.view.Surface.ROTATION_0)
                            .build()

                        val isProcessing = AtomicBoolean(false)

                        imageAnalysis.setAnalyzer(analyzerExecutor) { imageProxy ->
                            if (isProcessing.get()) {
                                imageProxy.close()
                                return@setAnalyzer
                            }
                            
                            val rotation = imageProxy.imageInfo.rotationDegrees
                            val srcBitmap: Bitmap? = try {
                                imageProxy.toBitmap()
                            } catch (e: Exception) {
                                Log.e(TAG, "toBitmap failed: ${e.message}")
                                null
                            }
                            imageProxy.close()
                            if (srcBitmap == null) return@setAnalyzer

                            isProcessing.set(true)
                            try {
                                if (!segmentor.isReady()) {
                                    kotlinx.coroutines.runBlocking { segmentor.initialize() }
                                }

                                val bitmap: Bitmap = if (rotation != 0) {
                                    val matrix = Matrix().apply { postRotate(rotation.toFloat()) }
                                    val rotated = Bitmap.createBitmap(
                                        srcBitmap, 0, 0,
                                        srcBitmap.width, srcBitmap.height, matrix, true
                                    )
                                    if (rotated !== srcBitmap) srcBitmap.recycle()
                                    rotated
                                } else {
                                    srcBitmap
                                }

                                val result: MaskData? = segmentor.runSegmentation(bitmap)

                                // 4. Create mask bitmap on background thread
                                var maskBmp: Bitmap? = null
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
                                    
                                    overlayView?.setMaskData(finalResult, maskBmp, bitmap.width, bitmap.height, 0)
                                    viewModel.updateLiveMask(finalResult, bitmap.width, bitmap.height)

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
                                    bitmap.recycle()
                                }
                            } catch (e: Exception) {
                                Log.e(TAG, "Analyzer error: ${e.message}")
                            } finally {
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
        "searching" -> Color(0xFF607D8B) to stringResource(R.string.auto_status_searching)
        "stabilizing" -> Color(0xFFFFA000) to stringResource(R.string.auto_status_stabilizing)
        "low_quality" -> Color(0xFFE53935) to stringResource(R.string.auto_status_low_quality)
        "ready" -> Color(0xFF43A047) to stringResource(R.string.auto_status_ready)
        "capturing" -> Color(0xFF6750A4) to stringResource(R.string.auto_status_capturing)
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
