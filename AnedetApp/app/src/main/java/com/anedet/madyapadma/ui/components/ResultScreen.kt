package com.anedet.madyapadma.ui.components

import android.graphics.BitmapFactory
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.tween
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.BorderStyle
import androidx.compose.material.icons.filled.Layers
import androidx.compose.material.icons.filled.LayersClear
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.scale
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.geometry.CornerRadius
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.anedet.madyapadma.R
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import com.anedet.madyapadma.camera.CameraViewModel
import com.anedet.madyapadma.model.PredictionResult

private enum class MaskMode { OFF, BORDER, FULL }

@Composable
private fun ProbabilityBar(
    value: Float,
    color: Color,
    modifier: Modifier = Modifier
) {
    val fill = (value * 100).coerceIn(5f, 100f)

    Box(
        modifier = modifier.background(Color(0xFFE0E0E0)),
        contentAlignment = Alignment.BottomCenter
    ) {
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .height(60.dp * (fill / 100f))
                .align(Alignment.BottomCenter)
                .background(color)
        )
    }
}

@Composable
fun ResultScreen(
    imagePath: String,
    onRetake: () -> Unit,
    viewModel: CameraViewModel = viewModel()
) {
    val predictionResult by viewModel.predictionResult.collectAsStateWithLifecycle()
    val isAnalyzing     by viewModel.isAnalyzing.collectAsStateWithLifecycle()

    LaunchedEffect(imagePath) {
        viewModel.analyzeImage(imagePath)
    }

    val isLoading = isAnalyzing || (predictionResult == null)

    val scaleAnim by animateFloatAsState(
        targetValue = if (!isLoading) 1f else 0.8f,
        animationSpec = tween(durationMillis = 500),
        label = "scale"
    )

    Surface(modifier = Modifier.fillMaxSize()) {
        when {
            isLoading -> {
                Box(
                    modifier = Modifier.fillMaxSize(),
                    contentAlignment = Alignment.Center
                ) {
                    Column(horizontalAlignment = Alignment.CenterHorizontally) {
                        CircularProgressIndicator(
                            modifier = Modifier.size(48.dp),
                            color = MaterialTheme.colorScheme.primary
                        )
                        Spacer(modifier = Modifier.height(16.dp))
                        Text(stringResource(R.string.analyzing), style = MaterialTheme.typography.bodyLarge)
                    }
                }
            }

            predictionResult != null -> {
                val prediction = predictionResult!!
                if (prediction.error != null) {
                    ErrorContent(message = prediction.error, onRetake = onRetake)
                } else {
                    ResultContent(
                        imagePath = imagePath,
                        prediction = prediction,
                        scaleAnim = scaleAnim,
                        onRetake = onRetake
                    )
                }
            }
        }
    }
}

@Composable
private fun ResultContent(
    imagePath: String,
    prediction: PredictionResult,
    scaleAnim: Float,
    onRetake: () -> Unit
) {
    var maskMode by remember { mutableStateOf(MaskMode.FULL) }
    val bitmap = remember(imagePath) { BitmapFactory.decodeFile(imagePath) }
    val isAnemic = prediction.isAnemic
    val diagColor = if (isAnemic) Color(0xFFE53935) else Color(0xFF43A047)
    val diagText = if (isAnemic) stringResource(R.string.anemic) else stringResource(R.string.non_anemic)

    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(16.dp),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Text(
            text = stringResource(R.string.diagnosis_result),
            style = MaterialTheme.typography.headlineMedium,
            fontWeight = FontWeight.Bold,
            modifier = Modifier.padding(top = 24.dp)
        )

        Spacer(modifier = Modifier.height(12.dp))

        Box(contentAlignment = Alignment.TopEnd) {
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .height(300.dp)
                    .clip(RoundedCornerShape(16.dp))
                    .scale(scaleAnim)
            ) {
                if (bitmap != null) {
                    Image(
                        bitmap = bitmap.asImageBitmap(),
                        contentDescription = "Captured eye",
                        modifier = Modifier.fillMaxSize(),
                        contentScale = ContentScale.Fit
                    )

                    when (maskMode) {
                        MaskMode.FULL -> {
                            if (prediction.maskOverlay != null) {
                                Image(
                                    bitmap = prediction.maskOverlay.asImageBitmap(),
                                    contentDescription = "Mask overlay",
                                    modifier = Modifier.fillMaxSize(),
                                    contentScale = ContentScale.Fit,
                                    alpha = 0.5f
                                )
                            }
                        }
                        MaskMode.BORDER -> {
                            val bbox = prediction.bbox
                            if (bbox != null) {
                                val bw = bitmap.width.toFloat()
                                val bh = bitmap.height.toFloat()
                                Canvas(modifier = Modifier.fillMaxSize()) {
                                    val sx = size.width / bw
                                    val sy = size.height / bh
                                    drawRoundRect(
                                        color = Color(0xFF4CAF50),
                                        topLeft = Offset(bbox.left * sx, bbox.top * sy),
                                        size = Size(bbox.width() * sx, bbox.height() * sy),
                                        cornerRadius = CornerRadius(12f, 12f),
                                        style = Stroke(width = 4f)
                                    )
                                }
                            }
                        }
                        MaskMode.OFF -> { }
                    }
                }
            }

            MaskToggleButton(
                mode = maskMode,
                onToggle = {
                    maskMode = when (maskMode) {
                        MaskMode.OFF -> MaskMode.BORDER
                        MaskMode.BORDER -> MaskMode.FULL
                        MaskMode.FULL -> MaskMode.OFF
                    }
                },
                modifier = Modifier.padding(8.dp)
            )
        }

        Spacer(modifier = Modifier.height(24.dp))

        Box(
            modifier = Modifier
                .fillMaxWidth()
                .clip(RoundedCornerShape(12.dp))
                .background(diagColor.copy(alpha = 0.15f))
                .padding(16.dp),
            contentAlignment = Alignment.Center
        ) {
            Column(horizontalAlignment = Alignment.CenterHorizontally) {
                Text(
                    text = diagText,
                    fontSize = 28.sp,
                    fontWeight = FontWeight.ExtraBold,
                    color = diagColor
                )
                Spacer(modifier = Modifier.height(8.dp))
                Text(
                    text = "Confidence: ${"%.1f".format(prediction.confidence * 100)}%",
                    style = MaterialTheme.typography.bodyLarge,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }
        }

        Spacer(modifier = Modifier.height(16.dp))

        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceEvenly,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Column(
                modifier = Modifier.weight(1f),
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                Text(
                    text = stringResource(R.string.anemic),
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.SemiBold,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
                Text(
                    text = "${"%.1f".format(prediction.anemicProbability * 100)}%",
                    style = MaterialTheme.typography.titleLarge,
                    fontWeight = FontWeight.Bold,
                    color = if (isAnemic) Color(0xFFE53935) else Color.Unspecified
                )
            }

            Spacer(modifier = Modifier.width(8.dp))

            ProbabilityBar(
                value = prediction.anemicProbability,
                color = Color(0xFFE53935),
                modifier = Modifier.width(6.dp).height(60.dp)
            )

            Spacer(modifier = Modifier.width(6.dp))

            ProbabilityBar(
                value = prediction.nonAnemicProbability,
                color = Color(0xFF43A047),
                modifier = Modifier.width(6.dp).height(60.dp)
            )

            Spacer(modifier = Modifier.width(8.dp))

            Column(
                modifier = Modifier.weight(1f),
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                Text(
                    text = stringResource(R.string.non_anemic),
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.SemiBold,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
                Text(
                    text = "${"%.1f".format(prediction.nonAnemicProbability * 100)}%",
                    style = MaterialTheme.typography.titleLarge,
                    fontWeight = FontWeight.Bold,
                    color = if (!isAnemic) Color(0xFF43A047) else Color.Unspecified
                )
            }
        }

        Spacer(modifier = Modifier.height(6.dp))

        Text(
            text = "Inference: ${prediction.inferenceTimeMs}ms",
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )

        Spacer(modifier = Modifier.height(20.dp))

        Button(
            onClick = onRetake,
            modifier = Modifier
                .fillMaxWidth()
                .height(56.dp),
            shape = RoundedCornerShape(12.dp),
            colors = ButtonDefaults.buttonColors(containerColor = MaterialTheme.colorScheme.primary)
        ) {
            Text(stringResource(R.string.retake), fontSize = 16.sp)
        }

        Spacer(modifier = Modifier.height(16.dp))
    }
}

@Composable
private fun MaskToggleButton(
    mode: MaskMode,
    onToggle: () -> Unit,
    modifier: Modifier = Modifier
) {
    val icon = when (mode) {
        MaskMode.FULL -> Icons.Filled.Layers
        MaskMode.BORDER -> Icons.Filled.BorderStyle
        MaskMode.OFF -> Icons.Filled.LayersClear
    }
    val desc = when (mode) {
        MaskMode.FULL -> "Full mask"
        MaskMode.BORDER -> "Border mask"
        MaskMode.OFF -> "No mask"
    }

    IconButton(
        onClick = onToggle,
        modifier = modifier
            .size(40.dp)
            .background(Color.Black.copy(alpha = 0.4f), CircleShape)
            .clip(CircleShape)
    ) {
        Icon(
            imageVector = icon,
            contentDescription = desc,
            tint = Color.White,
            modifier = Modifier.size(22.dp)
        )
    }
}

@Composable
private fun ErrorContent(
    message: String,
    onRetake: () -> Unit,
    modifier: Modifier = Modifier
) {
    Column(
        modifier = modifier.padding(32.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {
        Text(
            text = "Detection Failed",
            style = MaterialTheme.typography.headlineMedium,
            fontWeight = FontWeight.Bold,
            color = MaterialTheme.colorScheme.error
        )
        Spacer(modifier = Modifier.height(16.dp))
        Text(
            text = message,
            style = MaterialTheme.typography.bodyLarge,
            textAlign = TextAlign.Center,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )
        Spacer(modifier = Modifier.height(24.dp))
        Button(onClick = onRetake) { Text("Try Again") }
    }
}
