package com.anedet.madyapadma.ui

import android.net.Uri
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.DrawerValue
import androidx.compose.material3.DrawerState
import androidx.compose.material3.ModalNavigationDrawer
import androidx.compose.material3.rememberDrawerState
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.ui.Modifier
import androidx.core.os.LocaleListCompat
import androidx.appcompat.app.AppCompatDelegate
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.navigation.NavType
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import androidx.navigation.navArgument
import com.anedet.madyapadma.camera.CameraViewModel
import com.anedet.madyapadma.camera.CaptureScreen
import com.anedet.madyapadma.ui.components.AppDrawerContent
import com.anedet.madyapadma.ui.components.ResultScreen
import com.anedet.madyapadma.ui.components.SettingsScreen
import kotlinx.coroutines.launch

@Composable
fun AnedetNavGraph() {
    val navController = rememberNavController()
    val cameraViewModel: CameraViewModel = viewModel()
    val drawerState: DrawerState = rememberDrawerState(initialValue = DrawerValue.Closed)
    val scope = rememberCoroutineScope()

    val currentLanguage by cameraViewModel.settings.language.collectAsState()

    fun setLanguage(lang: String) {
        cameraViewModel.settings.setLanguage(lang)
        val locale = LocaleListCompat.forLanguageTags(lang)
        AppCompatDelegate.setApplicationLocales(locale)
    }

    ModalNavigationDrawer(
        drawerState = drawerState,
        drawerContent = {
            AppDrawerContent(
                currentLanguage = currentLanguage,
                onLanguageSelected = { lang ->
                    setLanguage(lang)
                    scope.launch { drawerState.close() }
                }
            )
        }
    ) {
        Box(modifier = Modifier.fillMaxSize()) {
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
                        onSettings = { navController.navigate("settings") },
                        onOpenDrawer = { scope.launch { drawerState.open() } }
                    )
                }
                composable("settings") {
                    SettingsScreen(
                        viewModel = cameraViewModel,
                        onBack = { navController.popBackStack() },
                        onOpenDrawer = { scope.launch { drawerState.open() } }
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
    }
}
