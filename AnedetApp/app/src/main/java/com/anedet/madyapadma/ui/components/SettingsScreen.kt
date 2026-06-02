package com.anedet.madyapadma.ui.components

import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ArrowBack
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Slider
import androidx.appcompat.app.AppCompatDelegate
import androidx.compose.material3.Button
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.core.os.LocaleListCompat
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.unit.dp
import com.anedet.madyapadma.R
import androidx.lifecycle.viewmodel.compose.viewModel
import com.anedet.madyapadma.camera.CameraViewModel

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun SettingsScreen(
    onBack: () -> Unit,
    viewModel: CameraViewModel = viewModel()
) {
    val autoCapture by viewModel.isAutoCaptureEnabled.collectAsState()
    val threshold by viewModel.confidenceThreshold.collectAsState()
    val saveToDevice by viewModel.saveToDevice.collectAsState()

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text(stringResource(R.string.settings)) },
                navigationIcon = {
                    IconButton(onClick = onBack) {
                        Icon(Icons.Default.ArrowBack, contentDescription = "Back")
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
            Text(stringResource(R.string.inference_settings), style = MaterialTheme.typography.titleMedium)

            Row(
                modifier = Modifier.fillMaxWidth().padding(vertical = 8.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(stringResource(R.string.smart_auto_capture), modifier = Modifier.weight(1f))
                Switch(checked = autoCapture, onCheckedChange = { viewModel.setAutoCapture(it) })
            }

            Text("${stringResource(R.string.confidence_threshold)}: ${(threshold * 100).toInt()}%", modifier = Modifier.padding(top = 16.dp))
            Slider(
                value = threshold,
                onValueChange = { viewModel.setConfidenceThreshold(it) },
                valueRange = 0.1f..0.9f
            )

            Spacer(modifier = Modifier.height(24.dp))
            Text(stringResource(R.string.storage), style = MaterialTheme.typography.titleMedium)
            Row(
                modifier = Modifier.fillMaxWidth().padding(vertical = 8.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(stringResource(R.string.save_to_device), modifier = Modifier.weight(1f))
                Switch(checked = saveToDevice, onCheckedChange = { viewModel.setSaveToDevice(it) })
            }

            Spacer(modifier = Modifier.height(24.dp))
            Text(stringResource(R.string.language), style = MaterialTheme.typography.titleMedium)
            Row(
                modifier = Modifier.fillMaxWidth().padding(vertical = 8.dp),
                horizontalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                Button(onClick = { setLanguage("en") }, modifier = Modifier.weight(1f)) { Text("EN") }
                Button(onClick = { setLanguage("in") }, modifier = Modifier.weight(1f)) { Text("ID") }
                Button(onClick = { setLanguage("th") }, modifier = Modifier.weight(1f)) { Text("TH") }
            }
        }
    }
}

private fun setLanguage(langCode: String) {
    val appLocale: LocaleListCompat = LocaleListCompat.forLanguageTags(langCode)
    AppCompatDelegate.setApplicationLocales(appLocale)
}
