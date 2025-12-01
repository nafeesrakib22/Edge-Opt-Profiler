import tensorflow as tf
import time
import os
import numpy as np

class ModelProfiler:
    def profile_model(self, model_path):
        # 1. Measure File Size
        size_bytes = os.path.getsize(model_path)
        size_mb = size_bytes / (1024 * 1024)

        # 2. Measure Latency
        try:
            latency_ms = self._measure_inference_time(model_path)
        except Exception as e:
            print(f"⚠️ Profiling error for {model_path}: {e}")
            latency_ms = 0.0

        return {
            "size_mb": round(size_mb, 4),
            "latency_ms": round(latency_ms, 4)
        }

    def _measure_inference_time(self, model_path, runs=10):
        # Load Interpreter
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        input_shape = input_details[0]['shape']
        input_type = input_details[0]['dtype']

        # Generate random dummy data
        dummy_input = np.random.random(input_shape).astype(input_type)

        # Warmup
        interpreter.set_tensor(input_details[0]['index'], dummy_input)
        interpreter.invoke()

        # Timing
        start_time = time.time()
        for _ in range(runs):
            interpreter.set_tensor(input_details[0]['index'], dummy_input)
            interpreter.invoke()
        end_time = time.time()

        # Calculate average time per run in milliseconds
        return ((end_time - start_time) / runs) * 1000