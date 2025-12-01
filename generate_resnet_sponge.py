import tensorflow as tf
import numpy as np
import os
import cv2

def create_resnet_demo():
    print("⬇️ Downloading CIFAR-10 Dataset...")
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()

    # --- CONFIGURATION ---
    TRAIN_SIZE = 20000 
    VAL_SIZE = 1000    
    IMG_SIZE = 64      # ResNet works better with slightly larger inputs
    # ---------------------

    print(f"🔄 Resizing {TRAIN_SIZE + VAL_SIZE} images to {IMG_SIZE}x{IMG_SIZE}...")
    print("   (This might take 1-2 minutes)...")
    
    def resize_data(data):
        resized = []
        for i, img in enumerate(data):
            # Resize 32x32 -> 64x64
            new_img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            resized.append(new_img)
        return np.array(resized).astype('float32') / 255.0

    X_train_large = resize_data(X_train[:TRAIN_SIZE])
    y_train_large = y_train[:TRAIN_SIZE]
    
    X_val_large = resize_data(X_test[:VAL_SIZE])
    y_val_large = y_test[:VAL_SIZE]

    print(f"🧠 Building ResNet50V2 (The 'Wet Sponge')...")
    
    # Load ResNet50V2 (Pre-trained on ImageNet)
    # This model has ~25 Million parameters (10x bigger than MobileNet)
    base_model = tf.keras.applications.ResNet50V2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False, 
        weights='imagenet'
    )
    base_model.trainable = False 

    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(256, activation='relu'), # Extra dense layer for capacity
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])

    # Train
    print("⚙️ Fine-tuning (This will take time due to model size)...")
    model.fit(X_train_large, y_train_large, epochs=5, batch_size=64)

    # Verify
    print("🔍 Verifying Baseline Accuracy...")
    loss, acc = model.evaluate(X_val_large, y_val_large, verbose=0)
    print(f"✅ Baseline Accuracy: {acc:.4f}")

    # Save
    output_dir = "demo_resnet"
    os.makedirs(output_dir, exist_ok=True)
    
    print("💾 Saving files (Expect a large ~90MB file)...")
    model.save(os.path.join(output_dir, "resnet_cifar.h5"))
    np.save(os.path.join(output_dir, "X_val.npy"), X_val_large)
    np.save(os.path.join(output_dir, "y_val.npy"), y_val_large)

    print(f"\n✅ DONE! 'Wet Sponge' model ready in '{output_dir}/'")

if __name__ == "__main__":
    create_resnet_demo()

