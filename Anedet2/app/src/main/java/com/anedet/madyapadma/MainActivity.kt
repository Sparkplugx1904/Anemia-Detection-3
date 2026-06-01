package com.anedet.madyapadma

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import com.anedet.madyapadma.ui.AnedetNavGraph
import com.anedet.madyapadma.ui.AnedetTheme

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            AnedetTheme {
                AnedetNavGraph()
            }
        }
    }
}
