package com.anedet.madyapadma.ml

/**
 * Exponential Moving Average untuk multi-frame classification smoothing.
 * 
 * Bug 4.2 Fix: Stabilize prediction results dengan smooth recent frames,
 * mengurangi flickering pada borderline cases (confidence 50-55%).
 * 
 * EMA formula: EMA[t] = alpha * value[t] + (1 - alpha) * EMA[t-1]
 * where alpha determines weight of recent values (higher = more responsive)
 * 
 * @param windowSize Equivalent window size for decay calculation (3-5 frames typical)
 */
class ExponentialMovingAverage(
    windowSize: Int = 4
) {
    // Alpha = 2 / (N + 1) for equivalent SMA window of N frames
    private val alpha: Float = 2.0f / (windowSize + 1)
    
    private var emaValue: Float? = null
    private var sampleCount: Int = 0
    
    /**
     * Update EMA dengan new value.
     * 
     * @param value New sample value
     * @return Current smoothed EMA value
     */
    fun update(value: Float): Float {
        val current = emaValue
        
        emaValue = if (current == null) {
            // First sample - initialize
            value
        } else {
            // Apply EMA formula
            alpha * value + (1 - alpha) * current
        }
        
        sampleCount++
        return emaValue!!
    }
    
    /**
     * Get current EMA value without updating.
     * Returns null if no samples yet.
     */
    fun current(): Float? = emaValue
    
    /**
     * Reset EMA to initial state.
     */
    fun reset() {
        emaValue = null
        sampleCount = 0
    }
    
    /**
     * Check if EMA is stabilized (has enough samples).
     * Recommendation: Wait for at least 2-3 samples before trusting EMA.
     */
    fun isStabilized(minSamples: Int = 3): Boolean = sampleCount >= minSamples
    
    companion object {
        /**
         * Helper untuk smooth classification probability pair.
         * Maintains separate EMA for each class probability.
         */
        class PairEMA(windowSize: Int = 4) {
            private val ema1 = ExponentialMovingAverage(windowSize)
            private val ema2 = ExponentialMovingAverage(windowSize)
            
            fun update(prob1: Float, prob2: Float): Pair<Float, Float> {
                return Pair(ema1.update(prob1), ema2.update(prob2))
            }
            
            fun current(): Pair<Float, Float>? {
                val v1 = ema1.current()
                val v2 = ema2.current()
                return if (v1 != null && v2 != null) Pair(v1, v2) else null
            }
            
            fun reset() {
                ema1.reset()
                ema2.reset()
            }
            
            fun isStabilized(minSamples: Int = 3): Boolean {
                return ema1.isStabilized(minSamples) && ema2.isStabilized(minSamples)
            }
        }
    }
}
