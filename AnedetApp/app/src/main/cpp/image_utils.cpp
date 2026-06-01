#include <android/log.h>

#define LOG_TAG "AnedetImgUtils"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)

// TODO: Implement RGBA_8888 → float32 NCHW conversion with NEON intrinsics
// TODO: Implement bilinear resize for mask upsampling
