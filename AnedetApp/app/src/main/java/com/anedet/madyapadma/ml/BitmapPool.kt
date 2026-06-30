package com.anedet.madyapadma.ml

import android.graphics.Bitmap
import android.util.Log
import java.lang.ref.SoftReference
import java.util.Collections

/**
 * Simple Bitmap pooling untuk reuse bitmap allocations.
 * 
 * Mengurangi allocation overhead pada operations seperti rotation atau preprocessing
 * yang membuat intermediate bitmaps dengan ukuran predictable.
 * 
 * Uses SoftReference untuk allow GC reclaim memory under pressure.
 */
class BitmapPool(private val maxPoolSize: Int = 5) {
    
    private val pool = Collections.synchronizedSet(
        mutableSetOf<SoftReference<Bitmap>>()
    )
    
    /**
     * Get bitmap dari pool yang match width, height, config.
     * Returns null jika tidak ada yang cocok.
     */
    fun get(width: Int, height: Int, config: Bitmap.Config): Bitmap? {
        synchronized(pool) {
            val iterator = pool.iterator()
            while (iterator.hasNext()) {
                val ref = iterator.next()
                val bitmap = ref.get()
                
                if (bitmap == null) {
                    // Reference cleared by GC
                    iterator.remove()
                    continue
                }
                
                if (!bitmap.isMutable) {
                    iterator.remove()
                    continue
                }
                
                // Check if bitmap can be reused
                if (canUseForInBitmap(bitmap, width, height, config)) {
                    iterator.remove()
                    bitmap.eraseColor(0) // Clear previous content
                    return bitmap
                }
            }
            return null
        }
    }
    
    /**
     * Return bitmap ke pool untuk reuse.
     */
    fun put(bitmap: Bitmap?) {
        if (bitmap == null || bitmap.isRecycled || !bitmap.isMutable) {
            return
        }
        
        synchronized(pool) {
            // Enforce pool size limit
            if (pool.size >= maxPoolSize) {
                // Remove oldest (arbitrary - Set doesn't maintain order, tapi limit tetap enforced)
                val toRemove = pool.firstOrNull()
                if (toRemove != null) {
                    pool.remove(toRemove)
                    toRemove.get()?.recycle()
                }
            }
            
            pool.add(SoftReference(bitmap))
        }
    }
    
    /**
     * Clear semua bitmaps di pool.
     */
    fun clear() {
        synchronized(pool) {
            pool.forEach { ref ->
                ref.get()?.recycle()
            }
            pool.clear()
        }
    }
    
    /**
     * Check if bitmap can be reused for target dimensions and config.
     * 
     * From Android 4.4 (API 19)+, we can reuse if byte size is sufficient.
     * Since minSdk = 31, we can safely use byte size comparison.
     */
    private fun canUseForInBitmap(
        candidate: Bitmap,
        targetWidth: Int,
        targetHeight: Int,
        targetConfig: Bitmap.Config
    ): Boolean {
        val targetByteCount = targetWidth * targetHeight * getBytesPerPixel(targetConfig)
        return candidate.allocationByteCount >= targetByteCount
    }
    
    private fun getBytesPerPixel(config: Bitmap.Config): Int {
        return when (config) {
            Bitmap.Config.ARGB_8888 -> 4
            Bitmap.Config.RGB_565, Bitmap.Config.ARGB_4444 -> 2
            Bitmap.Config.ALPHA_8 -> 1
            else -> 4 // Default to ARGB_8888
        }
    }
    
    companion object {
        private const val TAG = "BitmapPool"
    }
}
