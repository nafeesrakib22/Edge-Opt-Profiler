import tensorflow as tf
import os

model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, input_shape=(5,), activation='relu'),
    tf.keras.layers.Dense(1)
])
model.compile(optimizer='adam', loss='mse')
os.makedirs("inputs", exist_ok=True)
model.save("inputs/my_test_model.h5")
print("🎉 Created dummy model at inputs/my_test_model.h5")