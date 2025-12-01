import tensorflow as tf
import os
import zipfile
import shutil
from src.pruner import ModelPruner

class TensorFlowLoader:
    def __init__(self, upload_dir="inputs", output_dir="outputs"):
        self.upload_dir = upload_dir
        self.output_dir = output_dir
        os.makedirs(self.upload_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

    def process_pipeline(self, file_path, X_data=None, y_data=None):
        filename = os.path.basename(file_path)
        
        # --- HELPER TO GET CONVERTER ---
        def get_converter():
            if filename.endswith(".h5"):
                # Standard Loading (Works perfectly if generated in same env)
                model = tf.keras.models.load_model(file_path)
                return tf.lite.TFLiteConverter.from_keras_model(model)
            elif filename.endswith(".zip"):
                model_dir = self._unzip_and_find(file_path)
                return tf.lite.TFLiteConverter.from_saved_model(model_dir)
            else:
                raise ValueError("For optimization, please upload .h5 or .zip")

        results = {}

        # 1. Baseline
        print("Generating Baseline...")
        converter = get_converter()
        tflite_model = converter.convert()
        results['baseline'] = self._save(tflite_model, "baseline.tflite")

        # 2. Dynamic Quantization
        print("Generating Dynamic Quantization...")
        converter = get_converter()
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        results['dynamic'] = self._save(tflite_model, "dynamic.tflite")

        # 3. Float16
        print("Generating Float16...")
        converter = get_converter()
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        tflite_model = converter.convert()
        results['fp16'] = self._save(tflite_model, "fp16.tflite")

        # 4. CONDITIONAL PRUNING
        if X_data is not None and y_data is not None:
            print("Generating Pruned Model...")
            pruner = ModelPruner()
            pruned_path = os.path.join(self.output_dir, "pruned.tflite")
            try:
                pruner.prune_and_save(file_path, X_data, y_data, pruned_path)
                results['pruning'] = pruned_path
            except Exception as e:
                print(f"⚠️ Pruning failed: {e}")

        return results

    def _unzip_and_find(self, zip_path):
        extract_path = os.path.join(self.upload_dir, "temp_extracted")
        with zipfile.ZipFile(zip_path, 'r') as z: z.extractall(extract_path)
        for root, _, files in os.walk(extract_path):
            if "saved_model.pb" in files: return root
        raise ValueError("Invalid Zip")

    def _save(self, content, name):
        path = os.path.join(self.output_dir, name)
        with open(path, "wb") as f: f.write(content)
        return path