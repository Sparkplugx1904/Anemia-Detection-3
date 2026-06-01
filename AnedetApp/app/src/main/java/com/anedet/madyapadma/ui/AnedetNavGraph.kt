package com.anedet.madyapadma.ui

import android.net.Uri
import androidx.compose.runtime.Composable
import androidx.navigation.NavType
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import androidx.navigation.navArgument
import com.anedet.madyapadma.camera.CaptureScreen
import com.anedet.madyapadma.ui.components.ResultScreen

@Composable
fun AnedetNavGraph() {
    val navController = rememberNavController()

    NavHost(
        navController = navController,
        startDestination = "capture"
    ) {
        composable("capture") {
            CaptureScreen(
                onResult = { resultPath ->
                    val encoded = Uri.encode(resultPath)
                    navController.navigate("result/$encoded")
                }
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
