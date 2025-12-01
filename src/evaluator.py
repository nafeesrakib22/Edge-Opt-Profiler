import numpy as np
import tensorflow as tf

class AccuracyEngine:
    def get_accuracy(self, model_path, variant_name, X_test=None, y_test=None):
        """
        Calculates accuracy.
        If data is provided, runs real inference.
        If not, returns a simulated score based on the variant type.
        """
        if X_test is not None and y_test is not None:
            return self._calculate_real_accuracy(model_path, X_test, y_test)
        else:
            return self._simulate_accuracy(variant_name)

    def _simulate_accuracy(self, variant_name):
        # Fallback values if no data is provided
        if variant_name == 'baseline': return 1.00
        elif variant_name == 'fp16': return 0.999
        elif variant_name == 'dynamic': return 0.965
        return 0.95

    def _calculate_real_accuracy(self, model_path, X_test, y_test):
        # 1. Init Interpreter
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        input_index = input_details[0]['index']
        output_index = output_details[0]['index']
        
        # 2. Prepare Loop
        # We test up to 100 samples to keep the UI responsive
        test_samples = min(len(X_test), 100) 
        correct_predictions = 0
        
        for i in range(test_samples):
            # Prepare Input
            # Ensure input shape matches (e.g., add batch dimension)
            input_data = X_test[i]
            if len(input_data.shape) < len(input_details[0]['shape']):
                 input_data = np.expand_dims(input_data, axis=0)
            
            input_data = input_data.astype(input_details[0]['dtype'])
            
            # Run Inference
            interpreter.set_tensor(input_index, input_data)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_index)
            
            # 3. Determine Prediction Type (Regression vs Classification)
            output_shape = output_data.shape
            is_classification = output_shape[-1] > 1 

            if is_classification:
                # CLASSIFICATION (e.g., MNIST, CIFAR)
                # We want the index of the highest probability
                prediction = np.argmax(output_data[0])
                
                # Get the Truth Label safely
                truth = y_test[i]
                # If labels are One-Hot encoded (e.g. [0,0,1,0]), take argmax
                if isinstance(truth, np.ndarray) and truth.size > 1:
                    truth = np.argmax(truth)
                # If labels are in an array like [5], extract the number
                elif isinstance(truth, np.ndarray):
                    truth = truth.item()
                
                if prediction == int(truth):
                    correct_predictions += 1
            else:
                # REGRESSION (e.g., House Prices)
                # We compare the raw values
                prediction = output_data[0][0]
                truth = y_test[i]
                if isinstance(truth, np.ndarray):
                    truth = truth.item()
                    
                # Arbitrary threshold for "correctness" in regression
                if abs(prediction - truth) < 0.5:
                    correct_predictions += 1
                
        return correct_predictions / test_samples