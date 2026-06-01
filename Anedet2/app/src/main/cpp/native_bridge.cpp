#include <jni.h>
#include <android/log.h>

#define LOG_TAG "AnedetNative"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

extern "C" JNIEXPORT jfloatArray JNICALL
Java_com_anedet_madyapadma_native_1jni_NativeBridge_runSegmentation(
    JNIEnv *env, jobject thiz,
    jbyteArray imageBuffer, jint width, jint height) {

    LOGI("runSegmentation called: %dx%d", width, height);
    // TODO: Implement native segmentation post-processing
    jfloatArray result = env->NewFloatArray(1);
    jfloat temp[] = {0.0f};
    env->SetFloatArrayRegion(result, 0, 1, temp);
    return result;
}

extern "C" JNIEXPORT jfloatArray JNICALL
Java_com_anedet_madyapadma_native_1jni_NativeBridge_postProcessMask(
    JNIEnv *env, jobject thiz,
    jfloatArray rawMask, jint imgW, jint imgH) {

    LOGI("postProcessMask called: %dx%d", imgW, imgH);
    // TODO: Implement mask upsampling + polygon extraction
    jfloatArray result = env->NewFloatArray(1);
    jfloat temp[] = {0.0f};
    env->SetFloatArrayRegion(result, 0, 1, temp);
    return result;
}

extern "C" JNIEXPORT jfloatArray JNICALL
Java_com_anedet_madyapadma_native_1jni_NativeBridge_cropAndPreprocess(
    JNIEnv *env, jobject thiz,
    jbyteArray sourceImage, jfloat left, jfloat top,
    jfloat right, jfloat bottom, jint targetW, jint targetH) {

    LOGI("cropAndPreprocess called: bbox=[%.1f,%.1f,%.1f,%.1f] target=%dx%d",
         left, top, right, bottom, targetW, targetH);
    // TODO: Implement crop + resize + normalize
    jfloatArray result = env->NewFloatArray(1);
    jfloat temp[] = {0.0f};
    env->SetFloatArrayRegion(result, 0, 1, temp);
    return result;
}

extern "C" JNIEXPORT jfloatArray JNICALL
Java_com_anedet_madyapadma_native_1jni_NativeBridge_runClassification(
    JNIEnv *env, jobject thiz,
    jfloatArray inputBuffer) {

    LOGI("runClassification called");
    // TODO: Implement classification post-processing
    jfloatArray result = env->NewFloatArray(2);
    jfloat temp[] = {0.5f, 0.5f};
    env->SetFloatArrayRegion(result, 0, 2, temp);
    return result;
}
