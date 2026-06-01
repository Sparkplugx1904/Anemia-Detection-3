package com.anedet.madyapadma.ml;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.GpuDelegate;

public class InterpreterBundle {
    private Interpreter interpreter;
    private GpuDelegate gpuDelegate;

    public InterpreterBundle(Interpreter interpreter, GpuDelegate gpuDelegate) {
        this.interpreter = interpreter;
        this.gpuDelegate = gpuDelegate;
    }

    public Interpreter getInterpreter() { return interpreter; }

    public void close() {
        if (interpreter != null) { interpreter.close(); interpreter = null; }
        if (gpuDelegate != null) { gpuDelegate.close(); gpuDelegate = null; }
    }
}
