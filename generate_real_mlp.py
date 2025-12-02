import tensorflow as tf
import numpy as np
import pandas as pd
import os

def create_california_housing_demo():
    print("⬇️ Downloading Real California Housing Data...")
    # Load directly from Google's host to avoid needing Scikit-Learn
    train_url = "https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv"
    test_url = "https://download.mlcc.google.com/mledu-datasets/california_housing_test.csv"
    
    df_train = pd.read_csv(train_url)
    df_test = pd.read_csv(test_url)

    # Combine both datasets to create a larger pool, then split manually
    # This gives us more control over the split ratio (e.g. 80/20)
    full_df = pd.concat([df_train, df_test])
    
    # Shuffle the data to ensure random distribution
    full_df = full_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Separate Features and Labels
    X_full = full_df.drop('median_house_value', axis=1).values
    y_full = full_df['median_house_value'].values

    # --- CRITICAL: NORMALIZE DATA ---
    mean = X_full.mean(axis=0)
    std = X_full.std(axis=0)
    
    X_full = (X_full - mean) / std
    
    # Scale labels too (divide by 100,000 so the loss isn't huge numbers)
    y_full = y_full / 100000.0

    # Split into Train (80%) and Test (20%)
    split_index = int(len(full_df) * 0.8)
    
    X_train, X_test = X_full[:split_index], X_full[split_index:]
    y_train, y_test = y_full[:split_index], y_full[split_index:]
    
    print(f"   -> Training Samples: {len(X_train)}")
    print(f"   -> Testing Samples:  {len(X_test)}")

    # Save a slice for the user to upload (Validation Set)
    # We save a larger validation slice (500) for better accuracy measurement in the tool
    X_val_save = X_test[:500].astype(np.float32)
    y_val_save = y_test[:500].astype(np.float32)

    print("🧠 Building Deep MLP Model...")
    # A "Real" architecture usually looks like a funnel (Wide -> Narrow)
    model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(8,)), # 8 Census Features
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2), # Prevents overfitting on real data
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1) # Predicts Price (in $100k units)
    ])

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    # Train
    print("⚙️ Training on Census Data...")
    model.fit(X_train, y_train, epochs=30, batch_size=32, verbose=1)

    # Verify
    print("🔍 Verifying Baseline Accuracy...")
    loss, mae = model.evaluate(X_test, y_test, verbose=0)
    print(f"✅ Baseline Mean Absolute Error: ${mae * 100000:.0f} (avg error in price)")

    # Save
    output_dir = "demo_real_mlp"
    os.makedirs(output_dir, exist_ok=True)
    
    print("💾 Saving files...")
    model.save(os.path.join(output_dir, "california_housing.h5"))
    np.save(os.path.join(output_dir, "X_val.npy"), X_val_save)
    np.save(os.path.join(output_dir, "y_val.npy"), y_val_save)

    print(f"\n✅ DONE! Real MLP ready in '{output_dir}/'")

if __name__ == "__main__":
    create_california_housing_demo()