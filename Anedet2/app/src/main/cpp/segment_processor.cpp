#include <android/log.h>

#define LOG_TAG "AnedetSegProc"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)

// TODO: Implement mask activation (sigmoid), upsampling with bilinear NEON,
// polygon extraction, and top-K filtering for YOLO26 NMS-free output
