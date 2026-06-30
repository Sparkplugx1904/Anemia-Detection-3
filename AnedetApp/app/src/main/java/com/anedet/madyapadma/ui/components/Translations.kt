package com.anedet.madyapadma.ui.components

import androidx.compose.runtime.Composable
import androidx.compose.runtime.compositionLocalOf

private val EN = mapOf(
    "app_name" to "Anedet",
    "settings" to "Settings",
    "inference_settings" to "Inference Settings",
    "smart_auto_capture" to "Smart Auto-Capture",
    "auto_capture_description" to "Automatically capture when conjunctiva is stable and in focus",
    "confidence_threshold" to "Detection Threshold",
    "threshold_description" to "Minimum detection confidence to accept a frame",
    "stability_frames" to "Stability Frames",
    "stability_description" to "Consecutive stable frames before auto-capture",
    "sharpness_min" to "Sharpness",
    "sharpness_description" to "Minimum Laplacian variance to ensure image is not blurry",
    "language" to "Language",
    "save" to "Save",
    "saved_to_gallery" to "Saved to gallery",
    "analyzing" to "Analyzing…",
    "anemic" to "ANEMIC",
    "non_anemic" to "NON-ANEMIC",
    "diagnosis_result" to "Diagnosis Result",
    "retake" to "Retake Photo",
    "confidence" to "Confidence",
    "anemia_class" to "Anemia Class",
    "non_anemia_class" to "Non-Anemia Class",
    "diagnostic_class" to "Diagnostic Class",
    "margin" to "Margin",
    "crop_preview_label" to "Crop",
    "toggle_crop_preview" to "Toggle cropped conjunctiva preview",
    "toggle_mask" to "Toggle mask overlay",
    "low_confidence_warning" to "Low confidence — please retake",
    "auto_status_searching" to "Looking for conjunctiva…",
    "auto_status_stabilizing" to "Hold steady…",
    "auto_status_capturing" to "Capturing…",
    "auto_status_low_quality" to "Image too blurry, hold camera steady",
    "auto_status_ready" to "Ready"
)

private val ID = mapOf(
    "app_name" to "Anedet",
    "settings" to "Pengaturan",
    "inference_settings" to "Pengaturan Inferensi",
    "smart_auto_capture" to "Pengambilan Otomatis Pintar",
    "auto_capture_description" to "Ambil foto otomatis saat konjungtiva stabil dan fokus",
    "confidence_threshold" to "Ambang Deteksi",
    "threshold_description" to "Keyakinan deteksi minimum untuk menerima sebuah frame",
    "stability_frames" to "Frame Stabil",
    "stability_description" to "Jumlah frame stabil berturut-turut sebelum auto-capture",
    "sharpness_min" to "Ketajaman",
    "sharpness_description" to "Variansi Laplacian minimum agar gambar tidak buram",
    "language" to "Bahasa",
    "save" to "Simpan",
    "saved_to_gallery" to "Tersimpan di galeri",
    "analyzing" to "Menganalisis…",
    "anemic" to "ANEMIA",
    "non_anemic" to "TIDAK ANEMIA",
    "diagnosis_result" to "Hasil Diagnosis",
    "retake" to "Ambil Ulang Foto",
    "confidence" to "Keyakinan",
    "anemia_class" to "Kelas Anemia",
    "non_anemia_class" to "Kelas Non-Anemia",
    "diagnostic_class" to "Kelas Diagnosis",
    "margin" to "Margin",
    "crop_preview_label" to "Crop",
    "toggle_crop_preview" to "Alihkan pratinjau crop konjungtiva",
    "toggle_mask" to "Alihkan mask overlay",
    "low_confidence_warning" to "Keyakinan rendah — silakan ambil ulang",
    "auto_status_searching" to "Mencari konjungtiva…",
    "auto_status_stabilizing" to "Tahan sebentar…",
    "auto_status_capturing" to "Mengambil…",
    "auto_status_low_quality" to "Gambar terlalu buram, tahan kamera tetap",
    "auto_status_ready" to "Siap"
)

private val TH = mapOf(
    "app_name" to "Anedet",
    "settings" to "การตั้งค่า",
    "inference_settings" to "การตั้งค่าการอนุมาน",
    "smart_auto_capture" to "การจับภาพอัจฉริยะอัตโนมัติ",
    "auto_capture_description" to "จับภาพอัตโนมัติเมื่อเยื่อบุตาขาวนิ่งและอยู่ในโฟกัส",
    "confidence_threshold" to "เกณฑ์การตรวจจับ",
    "threshold_description" to "ค่าความเชื่อมั่นขั้นต่ำในการยอมรับเฟรม",
    "stability_frames" to "เฟรมที่นิ่ง",
    "stability_description" to "จำนวนเฟรมที่นิ่งต่อเนื่องก่อนจับภาพอัตโนมัติ",
    "sharpness_min" to "ความคมชัด",
    "sharpness_description" to "ค่าความแปรปรวน Laplacian ขั้นต่ำเพื่อให้ภาพไม่เบลอ",
    "language" to "ภาษา",
    "save" to "บันทึก",
    "saved_to_gallery" to "บันทึกลงแกลเลอรีแล้ว",
    "analyzing" to "กำลังวิเคราะห์…",
    "anemic" to "โลหิตจาง",
    "non_anemic" to "ไม่เป็นโลหิตจาง",
    "diagnosis_result" to "ผลการวินิจฉัย",
    "retake" to "ถ่ายภาพใหม่",
    "confidence" to "ความเชื่อมั่น",
    "anemia_class" to "คลาสโลหิตจาง",
    "non_anemia_class" to "คลาสไม่เป็นโลหิตจาง",
    "diagnostic_class" to "คลาสการวินิจฉัย",
    "margin" to "ส่วนต่าง",
    "crop_preview_label" to "ครอบตัด",
    "toggle_crop_preview" to "สลับการแสดงภาพตัวอย่างการครอบตัด",
    "toggle_mask" to "สลับการแสดงผลหน้ากาก",
    "low_confidence_warning" to "ความเชื่อมั่นต่ำ — กรุณาถ่ายใหม่",
    "auto_status_searching" to "กำลังค้นหาเยื่อบุตาขาว…",
    "auto_status_stabilizing" to "กรุณานิ่งไว้…",
    "auto_status_capturing" to "กำลังจับภาพ…",
    "auto_status_low_quality" to "ภาพเบลอเกินไป กรุณาถือกล้องให้นิ่ง",
    "auto_status_ready" to "พร้อม"
)

val LocalLang = compositionLocalOf { "en" }

@Composable
fun t(key: String): String {
    val lang = LocalLang.current
    return when (lang) {
        "id" -> ID[key] ?: EN[key] ?: key
        "th" -> TH[key] ?: EN[key] ?: key
        else -> EN[key] ?: key
    }
}

fun resolveString(lang: String, key: String): String {
    return when (lang) {
        "id" -> ID[key] ?: EN[key] ?: key
        "th" -> TH[key] ?: EN[key] ?: key
        else -> EN[key] ?: key
    }
}
