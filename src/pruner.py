import tensorflow as tf
import tensorflow_model_optimization as tfmot
import numpy as np
import os

class ModelPruner:
    def prune_and_save(self, model_path, X_data, y_data, output_path):
        print(f"✂️ Pruning model (Smart Schedule)...")
        
        # 1. Load & Unfreeze
        model = tf.keras.models.load_model(model_path)
        model.trainable = True 
        
        # 2. Smart Epoch Calculation
        # If data is small (<5000), we need MORE epochs to heal (15).
        # If data is large, 6 epochs is enough.
        num_samples = len(X_data)
        if num_samples < 5000:
            epochs = 15
            print(f"   -> Small dataset detected ({num_samples}). Boosting fine-tuning to 15 epochs.")
        else:
            epochs = 6
            print(f"   -> Large dataset detected. Using standard 6 epochs.")

        batch_size = 32
        end_step = np.ceil(num_samples / batch_size).astype(np.int32) * epochs

        # 3. GENTLE SCHEDULE (30% Sparsity)
        pruning_params = {
            'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
                initial_sparsity=0.0,
                final_sparsity=0.30, # Safe limit
                begin_step=0,
                end_step=end_step
            )
        }

        # 4. Apply Wrapper
        model_for_pruning = tfmot.sparsity.keras.prune_low_magnitude(model, **pruning_params)

        # 5. RECOMPILE WITH ADAM (Restored for stability)
        # We switched back to Adam (1e-5) because SGD was failing to converge
        # on the small MobileNet dataset.
        model_for_pruning.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
            loss='sparse_categorical_crossentropy', 
            metrics=['accuracy']
        )
        
        # 6. Fine-Tune
        callbacks = [
            tfmot.sparsity.keras.UpdatePruningStep(),
        ]

        model_for_pruning.fit(X_data, y_data, 
                              batch_size=batch_size, 
                              epochs=epochs, 
                              verbose=1,
                              callbacks=callbacks)

        # 7. Convert & Save
        model_for_export = tfmot.sparsity.keras.strip_pruning(model_for_pruning)
        
        converter = tf.lite.TFLiteConverter.from_keras_model(model_for_export)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()

        with open(output_path, 'wb') as f:
            f.write(tflite_model)
            
        return output_path