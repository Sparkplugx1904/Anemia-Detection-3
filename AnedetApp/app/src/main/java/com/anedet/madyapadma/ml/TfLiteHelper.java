package com.anedet.madyapadma.ml;

import org.tensorflow.lite.Interpreter;
import java.nio.MappedByteBuffer;

public class TfLiteHelper {

    private static final int CPU_THREADS = 4;

    public static InterpreterBundle createInterpreter(MappedByteBuffer modelBuffer) {
        Interpreter.Options options = new Interpreter.Options();
        options.setNumThreads(CPU_THREADS);
        options.setUseXNNPACK(true);
        return new InterpreterBundle(new Interpreter(modelBuffer, options), null);
    }
}
