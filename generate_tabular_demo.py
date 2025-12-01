import tensorflow as tf
import numpy as np
import os

def create_tabular_demo():
    print("🏠 Generating Synthetic House Price Data (Tabular/MLP)...")
    
    # 1. Generate Fake Tabular Data
    # Let's say we have 1,000 houses with 8 features each 
    # (e.g., Rooms, Age, Location, Tax, etc.)
    num_samples = 1000
    num_features = 8
    
    # Random input features (0 to 1)
    X_data = np.random.rand(num_samples, num_features).astype(np.float32)
    
    # Generate fake prices: Price = 3*Feature1 + 2*Feature2 + Noise
    # This ensures the model actually has a pattern to learn
    y_data = (3 * X_data[:, 0]) + (2 * X_data[:, 1]) + np.random.normal(0, 0.1, num_samples)
    y_data = y_data.astype(np.float32)

    # Split into Train/Test
    split = 800
    X_train, X_test = X_data[:split], X_data[split:]
    y_train, y_test = y_data[:split], y_data[split:]

    # Save a slice for validation upload
    X_val_save = X_test[:100]
    y_val_save = y_test[:100]

    print("🧠 Building MLP (Dense) Model...")
    # 2. Define Architecture: Multi-Layer Perceptron (MLP)
    # Input -> Dense -> Dense -> Output
    model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(num_features,)), # Expecting 8 features
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1) # Output is 1 single number (The Price)
    ])

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    # Train
    print("⚙️ Training...")
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

    # Check accuracy
    loss, mae = model.evaluate(X_test, y_test, verbose=0)
    print(f"✅ Baseline Mean Absolute Error: {mae:.4f}")

    # 3. Save Files
    output_dir = "demo_tabular"
    os.makedirs(output_dir, exist_ok=True)
    
    model_path = os.path.join(output_dir, "housing_model.h5")
    model.save(model_path)
    
    np.save(os.path.join(output_dir, "X_val.npy"), X_val_save)
    np.save(os.path.join(output_dir, "y_val.npy"), y_val_save)

    print(f"\n📦 DONE! Files ready in '{output_dir}/'")
    print("1. Model: housing_model.h5")
    print("2. Data: X_val.npy, y_val.npy")

if __name__ == "__main__":
    create_tabular_demo()