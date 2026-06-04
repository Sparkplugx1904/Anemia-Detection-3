package com.anedet.madyapadma.ui.components

import android.graphics.BitmapFactory
import android.widget.Toast
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
import androidx.compose.material.icons.filled.Save
import androidx.compose.material.icons.filled.Warning
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.scale
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import com.anedet.madyapadma.R
import com.anedet.madyapadma.camera.CameraViewModel
import com.anedet.madyapadma.model.PredictionResult
import kotlin.math.max
import kotlin.math.min

private enum class MaskMode { OFF, BORDER, FULL }

@Composable
fun ResultScreen(
    imagePath: String,
    onRetake: () -> Unit,
    viewModel: CameraViewModel = viewModel()
) {
    val predictionResult by viewModel.predictionResult.collectAsStateWithLifecycle()
    val isAnalyzing     by viewModel.isAnalyzing.collectAsStateWithLifecycle()
    val context         = LocalContext.current

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
                        onRetake = onRetake,
                        onSave = {
                            viewModel.saveResultToGallery(imagePath)
                            Toast.makeText(context, context.getString(R.string.saved_to_gallery), Toast.LENGTH_SHORT).show()
                        }
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
    onRetake: () -> Unit,
    onSave: () -> Unit
) {
    var maskMode by remember { mutableStateOf(MaskMode.FULL) }
    val bitmap = remember(imagePath) { BitmapFactory.decodeFile(imagePath) }
    val isAnemic = prediction.isAnemic
    val diagColor = if (isAnemic) Color(0xFFE53935) else Color(0xFF43A047)
    val diagText = if (isAnemic) stringResource(R.string.anemic) else stringResource(R.string.non_anemic)
    val isLowConfidence = prediction.confidence < 0.55f

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
                                    contentScale = ContentScale.Fit
                                )
                            }
                        }
                        MaskMode.BORDER -> {
                            val polygon = prediction.polygon
                            if (polygon.size >= 3) {
                                val bw = bitmap.width.toFloat()
                                val bh = bitmap.height.toFloat()
                                Canvas(modifier = Modifier.fillMaxSize()) {
                                    val s = min(size.width / bw, size.height / bh)
                                    val dispW = bw * s
                                    val dispH = bh * s
                                    val offX = (size.width - dispW) / 2f
                                    val offY = (size.height - dispH) / 2f
                                    val path = androidx.compose.ui.graphics.Path().apply {
                                        val first = polygon.first()
                                        moveTo(
                                            offX + first.x * s,
                                            offY + first.y * s
                                        )
                                        for (i in 1 until polygon.size) {
                                            lineTo(
                                                offX + polygon[i].x * s,
                                                offY + polygon[i].y * s
                                            )
                                        }
                                        close()
                                    }
                                    drawPath(
                                        path = path,
                                        color = Color(0xFFFFFF00),
                                        style = Stroke(width = 5f)
                                    )
                                }
                            }
                        }
                        MaskMode.OFF -> { }
                    }
                }
            }

            IconButton(
                onClick = {
                    maskMode = when (maskMode) {
                        MaskMode.OFF -> MaskMode.BORDER
                        MaskMode.BORDER -> MaskMode.FULL
                        MaskMode.FULL -> MaskMode.OFF
                    }
                },
                modifier = Modifier
                    .padding(8.dp)
                    .size(40.dp)
                    .background(Color.Black.copy(alpha = 0.5f), CircleShape)
                    .clip(CircleShape)
            ) {
                val icon = when (maskMode) {
                    MaskMode.FULL -> Icons.Filled.Layers
                    MaskMode.BORDER -> Icons.Filled.BorderStyle
                    MaskMode.OFF -> Icons.Filled.LayersClear
                }
                Icon(
                    imageVector = icon,
                    contentDescription = stringResource(R.string.toggle_mask),
                    tint = Color.White,
                    modifier = Modifier.size(22.dp)
                )
            }
        }

        if (isLowConfidence) {
            Spacer(modifier = Modifier.height(12.dp))
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .clip(RoundedCornerShape(8.dp))
                    .background(Color(0xFFFFA000).copy(alpha = 0.18f))
                    .padding(12.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                Icon(
                    imageVector = Icons.Filled.Warning,
                    contentDescription = null,
                    tint = Color(0xFFE65100),
                    modifier = Modifier.size(20.dp)
                )
                Spacer(modifier = Modifier.width(8.dp))
                Text(
                    text = stringResource(R.string.low_confidence_warning),
                    style = MaterialTheme.typography.bodySmall,
                    color = Color(0xFFE65100)
                )
            }
        }

        Spacer(modifier = Modifier.height(20.dp))

        // Diagnosis card
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .clip(RoundedCornerShape(16.dp))
                .background(diagColor.copy(alpha = 0.12f))
                .padding(20.dp),
            contentAlignment = Alignment.Center
        ) {
            Column(horizontalAlignment = Alignment.CenterHorizontally) {
                Text(
                    text = diagText,
                    fontSize = 32.sp,
                    fontWeight = FontWeight.ExtraBold,
                    color = diagColor
                )
                Spacer(modifier = Modifier.height(4.dp))
                Text(
                    text = "${(prediction.diagnosisPercent * 100).format1dp()}%",
                    fontSize = 22.sp,
                    fontWeight = FontWeight.Bold,
                    color = diagColor
                )
            }
        }

        Spacer(modifier = Modifier.height(20.dp))

        // Confidence + Diagnostic Class breakdown
        Text(
            text = stringResource(R.string.diagnostic_class),
            style = MaterialTheme.typography.titleMedium,
            fontWeight = FontWeight.SemiBold,
            modifier = Modifier.fillMaxWidth(),
            textAlign = TextAlign.Start
        )
        Spacer(modifier = Modifier.height(8.dp))

        ClassProbabilityRow(
            label = stringResource(R.string.anemia_class),
            value = prediction.anemicProbability,
            isWinner = isAnemic,
            color = Color(0xFFE53935)
        )
        Spacer(modifier = Modifier.height(8.dp))
        ClassProbabilityRow(
            label = stringResource(R.string.non_anemia_class),
            value = prediction.nonAnemicProbability,
            isWinner = !isAnemic,
            color = Color(0xFF43A047)
        )

        Spacer(modifier = Modifier.height(20.dp))

        // Confidence summary card
        Card(
            modifier = Modifier.fillMaxWidth(),
            shape = RoundedCornerShape(12.dp)
        ) {
            Column(modifier = Modifier.padding(16.dp)) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text(
                        text = stringResource(R.string.confidence),
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        modifier = Modifier.weight(1f)
                    )
                    Text(
                        text = "${(prediction.confidence * 100).format1dp()}%",
                        style = MaterialTheme.typography.titleLarge,
                        fontWeight = FontWeight.Bold,
                        color = if (isLowConfidence) Color(0xFFE65100) else diagColor
                    )
                }
                Spacer(modifier = Modifier.height(8.dp))
                ConfidenceBar(
                    confidence = prediction.confidence,
                    color = if (isLowConfidence) Color(0xFFE65100) else diagColor
                )
                Spacer(modifier = Modifier.height(8.dp))
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text(
                        text = stringResource(R.string.margin),
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        modifier = Modifier.weight(1f)
                    )
                    Text(
                        text = "${(prediction.margin * 100).format1dp()}%",
                        style = MaterialTheme.typography.bodyMedium,
                        fontWeight = FontWeight.Medium
                    )
                }
            }
        }

        Spacer(modifier = Modifier.height(8.dp))
        Text(
            text = "Inference: ${prediction.inferenceTimeMs}ms",
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )

        Spacer(modifier = Modifier.height(20.dp))

        // Action buttons: Retake + Save
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            OutlinedButton(
                onClick = onRetake,
                modifier = Modifier
                    .weight(1f)
                    .height(56.dp),
                shape = RoundedCornerShape(12.dp)
            ) {
                Text(stringResource(R.string.retake), fontSize = 14.sp)
            }
            Button(
                onClick = onSave,
                modifier = Modifier
                    .weight(1f)
                    .height(56.dp),
                shape = RoundedCornerShape(12.dp),
                colors = ButtonDefaults.buttonColors(containerColor = MaterialTheme.colorScheme.primary)
            ) {
                Icon(
                    imageVector = Icons.Filled.Save,
                    contentDescription = null,
                    modifier = Modifier.size(20.dp)
                )
                Spacer(modifier = Modifier.width(8.dp))
                Text(stringResource(R.string.save), fontSize = 14.sp)
            }
        }

        Spacer(modifier = Modifier.height(16.dp))
    }
}

@Composable
private fun ClassProbabilityRow(
    label: String,
    value: Float,
    isWinner: Boolean,
    color: Color
) {
    val pct = (value * 100).coerceIn(0f, 100f)
    val barWidth = (value * 100).coerceIn(2f, 100f)
    val bgColor = if (isWinner) color.copy(alpha = 0.10f) else Color(0xFFF5F5F5)
    val borderColor = if (isWinner) color else Color.Transparent
    val labelColor = if (isWinner) color else MaterialTheme.colorScheme.onSurface
    val labelWeight = if (isWinner) FontWeight.Bold else FontWeight.Medium

    Column(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(10.dp))
            .background(bgColor)
            .padding(12.dp)
    ) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Text(
                text = label,
                style = MaterialTheme.typography.bodyMedium,
                fontWeight = labelWeight,
                color = labelColor,
                modifier = Modifier.weight(1f)
            )
            Text(
                text = "${pct.format1dp()}%",
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.Bold,
                color = labelColor
            )
        }
        Spacer(modifier = Modifier.height(6.dp))
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .height(8.dp)
                .clip(RoundedCornerShape(4.dp))
                .background(Color(0xFFE0E0E0))
        ) {
            Box(
                modifier = Modifier
                    .fillMaxWidth(fraction = barWidth / 100f)
                    .height(8.dp)
                    .background(color)
            )
        }
    }
}

@Composable
private fun ConfidenceBar(
    confidence: Float,
    color: Color
) {
    val pct = (confidence * 100).coerceIn(2f, 100f)
    Box(
        modifier = Modifier
            .fillMaxWidth()
            .height(10.dp)
            .clip(RoundedCornerShape(5.dp))
            .background(Color(0xFFE0E0E0))
    ) {
        Box(
            modifier = Modifier
                .fillMaxWidth(fraction = pct / 100f)
                .height(10.dp)
                .background(color)
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

private fun Float.format1dp(): String = "%.1f".format(this)
