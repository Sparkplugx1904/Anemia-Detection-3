# AnedetApp ProGuard Rules
# CRITICAL: These rules prevent runtime crashes in release builds

# ═══════════════════════════════════════════════════════════════
# DEBUGGING & STACK TRACES
# ═══════════════════════════════════════════════════════════════
-keepattributes SourceFile,LineNumberTable
-renamesourcefileattribute SourceFile
-keepattributes Signature,InnerClasses,EnclosingMethod

# ═══════════════════════════════════════════════════════════════
# TENSORFLOW LITE - CRITICAL
# ═══════════════════════════════════════════════════════════════
-keep class org.tensorflow.lite.** { *; }
-keepclassmembers class org.tensorflow.lite.** { *; }
-keep class org.tensorflow.lite.gpu.** { *; }
-keep class org.tensorflow.lite.nnapi.** { *; }

# Keep model signatures (reflection-based loading)
-keepclassmembers class * {
    @org.tensorflow.lite.annotations.UsedByReflection *;
}

# ═══════════════════════════════════════════════════════════════
# JETPACK COMPOSE
# ═══════════════════════════════════════════════════════════════
-keep class androidx.compose.runtime.** { *; }
-keep @androidx.compose.runtime.Composable class * { *; }
-keep class androidx.compose.material3.** { *; }
-keep class androidx.compose.foundation.** { *; }
-keep class androidx.compose.ui.** { *; }

# Compose lambdas
-keepclassmembers class androidx.compose.** {
    <methods>;
}

# ═══════════════════════════════════════════════════════════════
# CAMERAX - CRITICAL
# ═══════════════════════════════════════════════════════════════
-keep class androidx.camera.** { *; }
-keepclassmembers class androidx.camera.** { *; }

# ImageCapture callbacks (called via reflection)
-keep class * implements androidx.camera.core.ImageCapture$OnImageSavedCallback { *; }
-keep class * implements androidx.camera.core.ImageCapture$OnImageCaptureCallback { *; }
-keep class * implements androidx.camera.core.ImageAnalysis$Analyzer { *; }

# CameraX lifecycle
-keep class * extends androidx.camera.lifecycle.ProcessCameraProvider { *; }

# ═══════════════════════════════════════════════════════════════
# KOTLIN & COROUTINES
# ═══════════════════════════════════════════════════════════════
-keepclassmembers class kotlinx.coroutines.** {
    volatile <fields>;
}
-keep class kotlinx.coroutines.** { *; }

# Kotlin metadata
-keep class kotlin.Metadata { *; }

# ═══════════════════════════════════════════════════════════════
# APP DATA MODELS - CRITICAL for Navigation
# ═══════════════════════════════════════════════════════════════
-keep class com.anedet.madyapadma.model.** { *; }
-keepclassmembers class com.anedet.madyapadma.model.** { *; }

# Parcelable (if used in future)
-keepclassmembers class * implements android.os.Parcelable {
    public static final ** CREATOR;
}

# ═══════════════════════════════════════════════════════════════
# VIEWMODEL & LIFECYCLE
# ═══════════════════════════════════════════════════════════════
-keep class * extends androidx.lifecycle.ViewModel { *; }
-keep class androidx.lifecycle.** { *; }

# SavedStateHandle (if used)
-keepclassmembers class androidx.lifecycle.SavedStateHandle {
    <init>(...);
}

# ═══════════════════════════════════════════════════════════════
# NAVIGATION COMPONENT
# ═══════════════════════════════════════════════════════════════
-keep class androidx.navigation.** { *; }
-keepclassmembers class androidx.navigation.** {
    <methods>;
}

# ═══════════════════════════════════════════════════════════════
# ANDROID SYSTEM
# ═══════════════════════════════════════════════════════════════
# Keep system services
-keep class android.hardware.** { *; }
-keep class android.graphics.** { *; }

# SharedPreferences
-keepclassmembers class * implements android.content.SharedPreferences$Editor {
    <methods>;
}

# ═══════════════════════════════════════════════════════════════
# R8 OPTIMIZATIONS
# ═══════════════════════════════════════════════════════════════
# Allow aggressive optimization
-allowaccessmodification
-repackageclasses

# Remove logging in release (optional - comment out if you need logs)
# -assumenosideeffects class android.util.Log {
#     public static *** d(...);
#     public static *** v(...);
#     public static *** i(...);
# }

# ═══════════════════════════════════════════════════════════════
# WARNINGS SUPPRESSION (Only if verified safe)
# ═══════════════════════════════════════════════════════════════
# Suppress warnings for known-safe issues
-dontwarn org.tensorflow.lite.**
-dontwarn javax.annotation.**