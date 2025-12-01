import tensorflow as tf
import numpy as np
import os
import cv2  # You might need: pip install opencv-python

def create_mobilenet_demo():
    print("⬇️ Downloading CIFAR-10 Dataset...")
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()

    # MobileNet requires images larger than 32x32. We will resize to 96x96.
    # To keep this fast for a demo, we will only use a small slice of data.
    LIMIT_TRAIN = 1000  # Train on just 1000 images for speed
    LIMIT_TEST = 100    # Validate on 100 images

    print(f"🔄 Resizing images to 96x96 (Required for MobileNetV2)...")
    
    def resize_data(data):
        resized = []
        for img in data:
            # Resize 32x32 -> 96x96
            new_img = cv2.resize(img, (96, 96))
            resized.append(new_img)
        return np.array(resized).astype('float32') / 255.0 # Normalize 0-1

    # Process subsets
    X_train_big = resize_data(X_train[:LIMIT_TRAIN])
    y_train_subset = y_train[:LIMIT_TRAIN]
    
    X_test_big = resize_data(X_test[:LIMIT_TEST])
    y_test_subset = y_test[:LIMIT_TEST]

    print("🧠 Building MobileNetV2 (Transfer Learning)...")
    
    # 1. Load the Base Model (Pre-trained on ImageNet)
    # include_top=False removes the 1000-class output layer
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(96, 96, 3),
        include_top=False, 
        weights='imagenet'
    )
    
    # Freeze base model (so we don't destroy pre-trained weights)
    base_model.trainable = False

    # 2. Add our own Classification Head
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(10, activation='softmax') # CIFAR has 10 classes
    ])

    model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])

    # 3. Fine-Tune
    print("⚙️ Fine-tuning on CIFAR-10...")
    model.fit(X_train_big, y_train_subset, epochs=5, batch_size=32)

    # --- VALIDATION ACCURACY CHECK ---
    print("🔍 Checking accuracy on NEW data (Validation Set)...")
    loss, val_acc = model.evaluate(X_test_big, y_test_subset, verbose=0)
    print(f"✅ Real Validation Accuracy: {val_acc:.4f} ({val_acc*100:.1f}%)")
    # ----------------------

    # 4. Save
    output_dir = "demo_mobilenet"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save Model (This will be ~9-10 MB, much bigger/realer than the others)
    model.save(os.path.join(output_dir, "mobilenet_cifar.h5"))
    
    # Save Validation Data
    np.save(os.path.join(output_dir, "X_val.npy"), X_test_big)
    np.save(os.path.join(output_dir, "y_val.npy"), y_test_subset)

    print(f"\n✅ DONE! Real-world model ready in '{output_dir}/'")

if __name__ == "__main__":
    create_mobilenet_demo()