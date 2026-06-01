package com.anedet.madyapadma.ml;

import org.tensorflow.lite.Interpreter;

import java.nio.MappedByteBuffer;

public class TfLiteHelper {

    public static Interpreter createInterpreter(MappedByteBuffer modelBuffer) {
        Interpreter.Options options = new Interpreter.Options();
        options.setNumThreads(4);
        return new Interpreter(modelBuffer, options);
    }
}
