# Anedet — Workflow & Agent Guide

## 1. Project Identity

| Atribut | Nilai |
|---|---|
| **Nama Aplikasi** | Anedet |
| **Package Name** | `com.anedet.madyapadma` |
| **Deskripsi** | Deteksi anemia via citra konjungtiva mata menggunakan pipeline 2-stage YOLO26 |
| **Pipeline** | **Segmentasi (YOLO26n-seg)** → crop → **Klasifikasi (YOLO26s-cls)** |
| **Target Platform** | Android 13–16 (API 33–36) |
| **Tech Stack** | Kotlin + C++ NDK, Jetpack Compose, CameraX, TFLite, Vulkan/NNAPI |
| **Deployment** | APK sideload / internal distribution |
| **Optimizer Training** | SGD (bukan MuSGD default YOLO26) |
| **Ultralytics Version** | 8.4.56 |

---

## 2. Model Assets (Sudah Siap — Tidak Perlu Retraining)

### Segmentasi — YOLO26n-seg
| File | Path |
|---|---|
| TFLite FP16 | `Anemia Segmentor #2/exports/conj-seg-yolo26n/yolo26n_seg_fp16.tflite` |
| ONNX FP32 | `Anemia Segmentor #2/exports/conj-seg-yolo26n/yolo26n_seg_fp32.onnx` |
| Metadata | `Anemia Segmentor #2/exports/conj-seg-yolo26n/metadata.yaml` |
| Best Weights | `Anemia Segmentor #2/runs/segment/train-conj26n-seg/weights/best.pt` |

| Detail | Nilai |
|---|---|
| Input size | 320×320 |
| Classes | 1 (conjunctiva) |
| Quantization | FP16 uniform (`half=True`) |
| Ukuran file | ~2.7 MB |
| Val mask mAP50 | 0.9776 |
| Test mask mAP50 | 0.9644 |
| Dataset | 762 gambar (588 train, 108 val, 66 test) |

### Klasifikasi — YOLO26s-cls
| File | Path |
|---|---|
| TFLite FP16 | `Anemia Classifier #2/exports/anemia-cls-yolo26s/yolo26s_cls_fp16.tflite` |
| ONNX FP32 | `Anemia Classifier #2/exports/anemia-cls-yolo26s/yolo26s_cls_fp32.onnx` |
| Metadata | `Anemia Classifier #2/exports/anemia-cls-yolo26s/metadata.yaml` |
| Best Weights | `Anemia Classifier #2/runs/classify/train-anemia26s-cls/weights/best.pt` |

| Detail | Nilai |
|---|---|
| Input size | 448×448 |
| Classes | 2 (anemic, non-anemic) |
| Quantization | FP16 uniform (`half=True`) |
| Test accuracy | 87.5% |
| Precision anemic | 81.48% |
| Precision non-anemic | 91.89% |

---

## 3. Android App Development Workflow

### Phase 1: Foundation
- [x] Gradle build system (AGP 9.1.1, Gradle 9.3.1, JDK 21)
- [x] AndroidManifest.xml (package, permissions)
- [x] Resource files (themes, colors, strings, icons)
- [ ] **Tambahkan dependensi utama di `build.gradle.kts` (app):**
  - CameraX (CameraController, Lifecycle, View)
  - TensorFlow Lite (`tensorflow-lite`, `tensorflow-lite-support`, `tensorflow-lite-gpu`)
  - Jetpack Compose (BOM, Material3, Navigation, CameraX Compose)
  - Kotlin Coroutines
  - OpenCV Android SDK (atau compile dari source via NDK)
- [ ] **Create folder structure:**
  ```
  app/src/main/java/com/anedet/madyapadma/
  ├── AnedetApp.kt              # Application class
  ├── MainActivity.kt           # Entry point
  ├── camera/
  │   ├── CameraScreen.kt       # CameraX composable
  │   └── CameraViewModel.kt
  ├── ml/
  │   ├── Segmentor.kt          # TFLite segmentation wrapper
  │   ├── Classifier.kt         # TFLite classification wrapper
  │   └── AnemiaPipeline.kt     # Orchestrates seg → crop → cls
  ├── model/
  │   ├── PredictionResult.kt   # Data class for results
  │   └── MaskData.kt           # Mask polygon data
  ├── ui/
  │   ├── CaptureScreen.kt      # Camera preview + capture button
  │   ├── ResultScreen.kt       # Show mask overlay + diagnosis
  │   ├── HistoryScreen.kt      # Past results (optional)
  │   └── components/           # Reusable composables
  └── native/
      ├── jni/                  # JNI interface (Kotlin → C++)
      └── cpp/                  # Native C++ source (via NDK)
  ```

### Phase 2: Camera (CameraX)
- [ ] Implement `CaptureScreen.kt` dengan CameraX + Jetpack Compose
- [ ] Gunakan `ImageCapture` untuk capture gambar resolusi tinggi
- [ ] Set `ResolutionSelector` agar sesuai kebutuhan model (320×320 untuk seg, 448×448 untuk cls)
- [ ] Simpan captured image ke `File` sementara untuk diproses
- [ ] **UI:** Tombol capture → loading indicator → navigasi ke ResultScreen

### Phase 3: C++ NDK — Post-Processing
- [ ] Setup CMakeLists.txt untuk build native C++
- [ ] Implement native functions via JNI:
  - `nativeRunSegmentation(imageBuffer, width, height)` → `float[]` (mask raw output)
  - `nativePostProcessMask(rawMask, imgW, imgH)` → `float[]` (polygon/mask overlay)
  - `nativeCropAndPreprocess(sourceImage, bbox)` → `float[]` (crop + resize + normalize)
  - `nativeRunClassification(imageBuffer)` → `float[]` (probabilities)
- [ ] Optimasi NEON:
  - Build dengan `-Ofast -flto -mfpu=neon -D__ARM_NEON`
  - Resize + normalize RGBA_8888 → float32 NCHW di native
  - Mask upsampling dengan bilinear interpolation NEON
  - Top-K filtering (YOLO26 NMS-free)

### Phase 4: ML Pipeline (Seg → Crop → Cls)
- [ ] Implement `Segmentor.kt`:
  - Load `yolo26n_seg_fp16.tflite` dari assets
  - Preprocess (resize 320×320, normalize)
  - Run inference via TFLite Interpreter with **Vulkan delegate** (fallback chain)
  - Parse output: bounding box + prototypical masks
  - Call native `postProcessMask()` untuk mendapatkan polygon
- [ ] Implement `Classifier.kt`:
  - Load `yolo26s_cls_fp16.tflite` dari assets
  - Preprocess (crop conjunctiva dari mask, resize 448×448, normalize)
  - Run inference
  - Parse output: softmax probabilities
- [ ] Implement `AnemiaPipeline.kt`:
  - `suspend fun analyze(image: Bitmap): PredictionResult`
  - Step 1: Segmentasi → dapat mask & bbox
  - Step 2: Crop sesuai mask/bbox
  - Step 3: Klasifikasi → dapat probabilitas
  - Step 4: Gabungkan hasil → `PredictionResult`

### Phase 5: UI — Result Screen (Jetpack Compose)
- [ ] Implement `ResultScreen.kt`:
  - Tampilkan gambar asli dengan **mask overlay** (conjunctiva disorot)
  - Label diagnosis: **ANEMIC** (merah) atau **NON-ANEMIC** (hijau)
  - Confidence score
  - Tombol "Retake" → kembali ke kamera
  - Tombol "Save Result" (optional)
- [ ] Animasi: hasil muncul dengan fade-in + scale animation
- [ ] Handle error state: "Gambar tidak jelas", "Konjungtiva tidak terdeteksi"

### Phase 6: GPU Delegate — Fallback Chain
- [ ] Implement **Vulkan delegate** sebagai primary:
  ```kotlin
  val compatList = CompatibilityList()
  val options = Interpreter.Options().apply {
      addDelegate(GpuDelegate(compatList.bestOptionsForThisDevice))
  }
  ```
- [ ] Fallback ke **NNAPI** jika Vulkan gagal
- [ ] Fallback terakhir ke **XNNPACK (CPU)**
- [ ] Logging: catat delegate mana yang aktif untuk debugging
- [ ] Latency target: **< 1 detik** total (seg + cls)

### Phase 7: Polish & Error Handling
- [ ] Loading states: shimmer effect saat inferensi berjalan
- [ ] Error handling:
  - Tidak ada konjungtiva terdeteksi → pesan jelas + saran
  - Confidence rendah → tampilkan "Tidak Yakin" + saran retake
  - Gagal load model → fallback + log error
- [ ] Permissions: CAMERA, READ/WRITE_EXTERNAL_STORAGE (jika perlu)
- [ ] ProGuard rules (jika ada library yang perlu keep)
- [ ] Dark theme support (dari DayNight theme yang sudah ada)

### Phase 8: Testing & Build
- [ ] **Unit Test:**
  - `SegmentorTest` — mock interpreter, test pre/post processing
  - `ClassifierTest` — mock interpreter, test probability parsing
  - `AnemiaPipelineTest` — test orchestration logic
  - `PredictionResultTest` — data class validation
- [ ] **Instrumented Test:**
  - CameraX test dengan `CameraXTestUtil`
  - UI test dengan Compose UI Test + Espresso
  - Integrasi: real model + test image → validasi output
- [ ] **Build:**
  - `./gradlew assembleDebug` → APK debug
  - `./gradlew assembleRelease` → APK release (signed)
  - ProGuard di release variant
- [ ] **Manual Testing:**
  - Test di 3–5 device berbeda (minimal Snapdragon + Mali GPU)
  - Verify fallback chain bekerja (force-disable Vulkan via dev options)

---

## 4. Key Architecture Decisions

| Keputusan | Alasan |
|---|---|
| **Capture → Analyze → Result** (bukan real-time) | Sederhana, lebih stabil, tidak perlu 30 FPS, latensi < 1 detik acceptable |
| **Vulkan → NNAPI → XNNPACK** sebagai fallback chain | Vulkan lebih portable, NNAPI fastest jika compatible, XNNPACK sebagai jaring pengaman |
| **C++ NDK** untuk post-processing | Mask activation + upsampling berat di CPU; NEON optimization critical untuk < 1 detik |
| **Jetpack Compose** untuk UI | Modern, deklaratif, lebih sedikit boilerplate daripada XML |
| **CameraX** | Lifecycle-aware, simpler API daripada Camera2, built-in Compose support |
| **minSdk 33, targetSdk 36** | Fokus ke device modern; API 33+ punya permission model lebih baik |
| **FP16 uniform quantization** | Minimal accuracy loss (~0.1-0.2 mAP), 2× speedup di GPU, tidak crash di NNAPI |
| **SGD optimizer** (bukan MuSGD) | Sesuai training yang sudah dilakukan |
| **Ukuran model ~2.7 MB** | Cukup kecil untuk di-deploy via APK |

---

## 5. Checklist & Milestones

### 🟢 Phase 1: Foundation
- [ ] Tambah dependensi TFLite, CameraX, Compose, OpenCV
- [ ] Setup folder structure (packages)
- [ ] Verify build succeeds (`./gradlew assembleDebug`)

### 🔵 Phase 2: Camera
- [ ] CameraX preview visible
- [ ] ImageCapture menghasilkan file
- [ ] Navigasi ke ResultScreen setelah capture

### 🟡 Phase 3: C++ NDK
- [ ] CMakeLists.txt build success
- [ ] JNI bridge berfungsi
- [ ] Native post-processing menghasilkan mask polygon

### 🟠 Phase 4: ML Pipeline
- [ ] Segmentor → output bounding box + mask
- [ ] Classifier → output probability
- [ ] `analyze()` selesai dalam < 1 detik

### 🔴 Phase 5: UI
- [ ] ResultScreen menampilkan mask overlay
- [ ] Label diagnosis benar (ANEMIC / NON-ANEMIC)
- [ ] Animasi dan error handling rapi

### 🟣 Phase 6: GPU Delegate
- [ ] Vulkan delegate active di device compatible
- [ ] NNAPI fallback berfungsi
- [ ] XNNPACK fallback berfungsi

### ⚪ Phase 7: Polish
- [ ] Loading states
- [ ] Error handling untuk semua edge case
- [ ] Permission handling

### ⚫ Phase 8: Testing & Build
- [ ] Unit test passing
- [ ] Instrumented test passing
- [ ] APK debug & release build success
- [ ] Tested on 3+ physical devices
