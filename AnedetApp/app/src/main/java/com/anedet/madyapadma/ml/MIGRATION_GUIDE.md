# Panduan Migrasi AnedetApp — Perubahan Build & File

## 1. File yang harus diganti

| File lama (path di project) | File baru | Keterangan |
|---|---|---|
| `app/src/main/java/.../ml/TfLiteHelper.java` | `TfLiteHelper.kt` | Rewrite ke Kotlin + GPU delegate |
| `app/src/main/java/.../ml/Segmentor.kt` | `Segmentor.kt` | Letterbox + fix proto tensor 4D |
| `app/src/main/java/.../ml/Classifier.kt` | `Classifier.kt` | Preprocessing pipeline v2 |
| `app/src/main/java/.../ml/AnemiaPipeline.kt` | `AnemiaPipeline.kt` | Direct Bitmap crop + efficient overlay |
| `app/src/main/java/.../model/MaskData.kt` | `MaskData.kt` | Tambah proto space params |
| `app/src/main/java/.../ui/components/ResultScreen.kt` | `ResultScreen.kt` | collectAsStateWithLifecycle fix |

## 2. File yang harus DIHAPUS (dead code berbahaya)

```
app/src/main/java/.../native/jni/NativeBridge.kt
app/src/main/cpp/native_bridge.cpp
app/src/main/cpp/segment_processor.cpp
app/src/main/cpp/image_utils.cpp
app/src/main/cpp/CMakeLists.txt
```

Juga hapus dari `build.gradle.kts` (app-level) jika ada:
```kotlin
// HAPUS blok ini jika ada:
externalNativeBuild {
    cmake { path("src/main/cpp/CMakeLists.txt") }
}
```

## 3. Perubahan build.gradle.kts (app-level)

### Tambah dependency GPU delegate (sudah ada tflite-gpu, tapi pastikan tidak di-exclude):

```kotlin
// Pastikan baris ini ADA dan TIDAK ada exclude terlalu agresif:
implementation(libs.tflite.gpu)
// Jika sebelumnya ada: exclude(group = "org.tensorflow", module = "tensorflow-lite")
// Itu BOLEH dibiarkan karena mencegah duplikasi, tapi jangan exclude tflite-gpu-delegate-metadata
```

### Tambah dependency lifecycle untuk collectAsStateWithLifecycle:
```kotlin
implementation(libs.lifecycle.runtime.compose)  // sudah ada di versi sebelumnya ✓
```

### Hapus CMake jika ada:
```kotlin
// android { ... }
// HAPUS jika ada:
// externalNativeBuild { cmake { path = file("src/main/cpp/CMakeLists.txt") } }
```

## 4. Perubahan libs.versions.toml

```toml
# Perbaiki versi Kotlin (2.2.10 kemungkinan typo):
kotlin = "2.1.0"

# Update Compose BOM ke versi yang lebih baru:
composeBom = "2025.04.00"

# TFLite — versi 2.16.1 sudah ok, tapi GPU delegate support lebih baik di 2.17+:
tflite = "2.17.0"
```

## 5. Verifikasi setelah build

Cek logcat dengan filter `TfLiteHelper`:
- `GPU delegate aktif` → GPU berhasil diinisialisasi ✓
- `GPU delegate tidak didukung, pakai CPU 4 threads` → fallback CPU (normal untuk emulator)
- `GPU delegate gagal diinisialisasi, fallback ke CPU` → ada masalah driver GPU

Cek logcat setelah capture:
- Pastikan tidak ada `NullPointerException` dari `maskOverlay`
- Waktu inferensi seharusnya turun dari ~1500ms ke ~200-400ms dengan GPU aktif

## 6. Catatan penting Classifier

Model classifier sekarang menerima `Bitmap` langsung (bukan path).
`Classifier.classify(bitmap: Bitmap)` — pastikan tidak ada kode lain yang masih
memanggil versi lama `Classifier.classify(imagePath: String)`.

Preprocessing pipeline v2 di Classifier murni Kotlin — tidak butuh OpenCV.
Trade-off: BilateralFilter approx 3×3 lebih cepat dari OpenCV bilateral penuh (d=9),
tapi hasil masih cukup baik untuk preprocessing sebelum klasifikasi.
Jika hasil kurang optimal, bisa tambahkan OpenCV dependency dan ganti fungsi
`bilateralFilterApprox()` dengan panggilan ke OpenCV `Imgproc.bilateralFilter()`.
