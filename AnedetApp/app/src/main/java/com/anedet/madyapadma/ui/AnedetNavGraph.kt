package com.anedet.madyapadma.ui

import android.net.Uri
import androidx.compose.runtime.Composable
import androidx.navigation.NavType
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import androidx.navigation.navArgument
import androidx.lifecycle.viewmodel.compose.viewModel
import com.anedet.madyapadma.camera.CameraViewModel
import com.anedet.madyapadma.camera.CaptureScreen
import com.anedet.madyapadma.ui.components.ResultScreen
import com.anedet.madyapadma.ui.components.SettingsScreen

@Composable
fun AnedetNavGraph() {
    val navController = rememberNavController()
    val cameraViewModel: CameraViewModel = viewModel()

    NavHost(
        navController = navController,
        startDestination = "capture"
    ) {
        composable("capture") {
            CaptureScreen(
                viewModel = cameraViewModel,
                onResult = { resultPath ->
                    val encoded = Uri.encode(resultPath)
                    navController.navigate("result/$encoded")
                },
                onSettings = {
                    navController.navigate("settings")
                }
            )
        }
        composable("settings") {
            SettingsScreen(
                viewModel = cameraViewModel,
                onBack = { navController.popBackStack() }
            )
        }
        composable(
            route = "result/{imagePath}",
            arguments = listOf(navArgument("imagePath") { type = NavType.StringType })
        ) { backStackEntry ->
            val imagePath = Uri.decode(backStackEntry.arguments?.getString("imagePath") ?: "")
            ResultScreen(
                imagePath = imagePath,
                onRetake = { navController.popBackStack() }
            )
        }
    }
}
