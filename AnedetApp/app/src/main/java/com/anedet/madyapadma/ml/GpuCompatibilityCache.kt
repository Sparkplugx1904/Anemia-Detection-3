package com.anedet.madyapadma.ml

import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate

/**
 * Singleton cache untuk GPU compatibility check.
 * Menghindari repeated CompatibilityList() query yang memakan 30-50ms per call.
 * 
 * GPU compatibility adalah device-specific dan tidak berubah selama app lifecycle,
 * sehingga safe untuk di-cache.
 */
object GpuCompatibilityCache {
    
    private var cachedCompatibilityList: CompatibilityList? = null
    private var cachedIsSupported: Boolean? = null
    
    /**
     * Get CompatibilityList instance (cached after first call).
     */
    fun getCompatibilityList(): CompatibilityList {
        if (cachedCompatibilityList == null) {
            cachedCompatibilityList = CompatibilityList()
        }
        return cachedCompatibilityList!!
    }
    
    /**
     * Check if GPU delegate is supported on this device (cached).
     */
    fun isGpuSupported(): Boolean {
        if (cachedIsSupported == null) {
            cachedIsSupported = getCompatibilityList().isDelegateSupportedOnThisDevice
        }
        return cachedIsSupported!!
    }
    
    /**
     * Create GPU delegate with best options for this device.
     * Returns null if GPU not supported.
     */
    fun createGpuDelegateIfSupported(): GpuDelegate? {
        return if (isGpuSupported()) {
            GpuDelegate(getCompatibilityList().bestOptionsForThisDevice)
        } else {
            null
        }
    }
    
    /**
     * Get optimal CPU thread count for this device.
     * Fallback strategy ketika GPU tidak tersedia.
     */
    fun getOptimalCpuThreads(): Int {
        val availableProcessors = Runtime.getRuntime().availableProcessors()
        // Low-end devices (dual-core): use 2 threads
        // Mid-range (quad-core): use 3 threads (leave 1 for UI)
        // High-end (6+ cores): use 4 threads
        return when {
            availableProcessors <= 2 -> 2
            availableProcessors <= 4 -> 3
            else -> 4
        }
    }
}
