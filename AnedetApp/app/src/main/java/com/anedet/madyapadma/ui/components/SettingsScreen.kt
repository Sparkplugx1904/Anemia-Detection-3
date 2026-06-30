package com.anedet.madyapadma.ui.components

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.Menu
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Slider
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.lifecycle.viewmodel.compose.viewModel
import com.anedet.madyapadma.camera.CameraViewModel
import com.anedet.madyapadma.ui.components.t

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun SettingsScreen(
    onBack: () -> Unit,
    onOpenDrawer: () -> Unit,
    viewModel: CameraViewModel = viewModel()
) {
    val autoCapture   by viewModel.settings.smartAutoCapture.collectAsState()
    val threshold     by viewModel.settings.confidenceThreshold.collectAsState()
    val stability     by viewModel.settings.stabilityFrames.collectAsState()
    val sharpness     by viewModel.settings.sharpnessMin.collectAsState()

    var pendingAutoCapture by remember { mutableStateOf(autoCapture) }
    var pendingThreshold by remember { mutableStateOf(threshold) }
    var pendingStability by remember { mutableStateOf(stability) }
    var pendingSharpness by remember { mutableStateOf(sharpness) }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text(t("settings")) },
                navigationIcon = {
                    IconButton(onClick = onOpenDrawer) {
                        Icon(Icons.Default.Menu, contentDescription = "Open menu")
                    }
                },
                actions = {
                    IconButton(onClick = onBack) {
                        Icon(Icons.AutoMirrored.Filled.ArrowBack, contentDescription = "Back")
                    }
                }
            )
        }
    ) { padding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(padding)
                .padding(16.dp)
        ) {
            // Auto-Capture group
            SettingsGroup(t("smart_auto_capture")) {
                SwitchRow(
                    title = t("smart_auto_capture"),
                    subtitle = t("auto_capture_description"),
                    checked = pendingAutoCapture,
                    onCheckedChange = { pendingAutoCapture = it }
                )
            }

            Spacer(modifier = Modifier.height(16.dp))

            // Detection settings
            SettingsGroup(t("inference_settings")) {
                SliderRow(
                    title = t("confidence_threshold"),
                    subtitle = t("threshold_description"),
                    value = pendingThreshold,
                    onValueChange = { pendingThreshold = it },
                    valueRange = 0.10f..0.90f,
                    valueLabel = { "${(it * 100).toInt()}%" }
                )

                SliderRow(
                    title = t("stability_frames"),
                    subtitle = t("stability_description"),
                    value = pendingStability.toFloat(),
                    onValueChange = { pendingStability = it.toInt() },
                    valueRange = 2f..10f,
                    steps = 8,
                    valueLabel = { "${it.toInt()}" }
                )

                SliderRow(
                    title = t("sharpness_min"),
                    subtitle = t("sharpness_description"),
                    value = pendingSharpness,
                    onValueChange = { pendingSharpness = it },
                    valueRange = 1f..100f,
                    valueLabel = { "%.1f".format(it) }
                )
            }

            Spacer(modifier = Modifier.weight(1f))

            Button(
                onClick = {
                    viewModel.settings.setSmartAutoCapture(pendingAutoCapture)
                    viewModel.settings.setConfidenceThreshold(pendingThreshold)
                    viewModel.settings.setStabilityFrames(pendingStability)
                    viewModel.settings.setSharpnessMin(pendingSharpness)
                    onBack()
                },
                modifier = Modifier.align(Alignment.End)
            ) {
                Text(t("save"))
            }
        }
    }
}

@Composable
private fun SettingsGroup(title: String, content: @Composable () -> Unit) {
    Text(
        text = title,
        style = MaterialTheme.typography.titleMedium,
        fontWeight = FontWeight.SemiBold,
        color = MaterialTheme.colorScheme.primary,
        modifier = Modifier.padding(bottom = 8.dp)
    )
    Card(
        modifier = Modifier.fillMaxWidth(),
        elevation = CardDefaults.cardElevation(defaultElevation = 1.dp)
    ) {
        Column(modifier = Modifier.padding(8.dp)) { content() }
    }
}

@Composable
private fun SwitchRow(
    title: String,
    subtitle: String,
    checked: Boolean,
    onCheckedChange: (Boolean) -> Unit
) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 8.dp, horizontal = 8.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        Column(modifier = Modifier.weight(1f)) {
            Text(title, style = MaterialTheme.typography.bodyLarge, fontWeight = FontWeight.Medium)
            Text(
                subtitle,
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
        }
        Switch(checked = checked, onCheckedChange = onCheckedChange)
    }
}

@Composable
private fun SliderRow(
    title: String,
    subtitle: String,
    value: Float,
    onValueChange: (Float) -> Unit,
    valueRange: ClosedFloatingPointRange<Float>,
    steps: Int = 0,
    valueLabel: (Float) -> String
) {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 8.dp, horizontal = 8.dp)
    ) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Column(modifier = Modifier.weight(1f)) {
                Text(title, style = MaterialTheme.typography.bodyLarge, fontWeight = FontWeight.Medium)
                Text(
                    subtitle,
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }
            Text(
                text = valueLabel(value),
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.Bold,
                color = MaterialTheme.colorScheme.primary
            )
        }
        Slider(
            value = value,
            onValueChange = onValueChange,
            valueRange = valueRange,
            steps = steps
        )
    }
}
