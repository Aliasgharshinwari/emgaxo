import os
import sys
import logging

import tensorflow as tf
import numpy as np
import onnx
import tf2onnx

# ──────────────────────────────────────────────────────────────────────
# 1) Logging setup
# ──────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# 2) GPU configuration
# ──────────────────────────────────────────────────────────────────────
def configure_gpu():
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        logger.warning("No GPU detected; training will run on CPU.")
        return
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logger.info(f"Enabled memory growth on {len(gpus)} GPU(s).")
    except RuntimeError as e:
        logger.warning(f"Could not set GPU memory growth: {e}")

# ──────────────────────────────────────────────────────────────────────
# 3) Data loading with error handling
# ──────────────────────────────────────────────────────────────────────
def load_data():
    try:
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        x_train = x_train.astype('float32') / 255.0
        x_test  = x_test.astype('float32') / 255.0
        x_train = x_train.reshape(-1, 28*28)
        x_test  = x_test.reshape(-1, 28*28)
        return (x_train, y_train), (x_test, y_test)
    except Exception as e:
        logger.error(f"Failed to load MNIST data: {e}")
        sys.exit(1)

# ──────────────────────────────────────────────────────────────────────
# 4) Model definition
# ──────────────────────────────────────────────────────────────────────
def build_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(784,)),
        tf.keras.layers.Dense(256, activation='relu'),  # Layer 1
        tf.keras.layers.Dense(128, activation='relu'),  # Layer 2
        tf.keras.layers.Dense(64, activation='relu'),   # Layer 3
        tf.keras.layers.Dense(32, activation='relu'),   # Layer 4
        tf.keras.layers.Dense(16, activation='relu'),   # Layer 5
        tf.keras.layers.Dense(10, activation='softmax') # Output layer
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# ──────────────────────────────────────────────────────────────────────
# 5) Training with callbacks
# ──────────────────────────────────────────────────────────────────────
def train_model(model, x_train, y_train, x_val, y_val):
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(patience=2, factor=0.5),
        tf.keras.callbacks.ModelCheckpoint(
            filepath="models/best_model.h5",
            save_best_only=True,
            verbose=1
        ),
    ]
    try:
        history = model.fit(
            x_train, y_train,
            epochs=20,
            batch_size=128,
            validation_data=(x_val, y_val),
            callbacks=callbacks
        )
        return history
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)

# ──────────────────────────────────────────────────────────────────────
# 6) ONNX conversion with error handling
# ──────────────────────────────────────────────────────────────────────
def convert_to_onnx(model, export_path):
    try:
        os.makedirs(os.path.dirname(export_path), exist_ok=True)
        spec = (tf.TensorSpec((None, 784), tf.float32),)
        model_proto, external = tf2onnx.convert.from_keras(
            model,
            input_signature=spec,
            output_path=export_path
        )
        logger.info(f"Successfully exported ONNX model to: {export_path}")
        return model_proto
    except Exception as e:
        logger.error(f"ONNX conversion failed: {e}")
        return None

# ──────────────────────────────────────────────────────────────────────
# 7) Main entrypoint
# ──────────────────────────────────────────────────────────────────────
def main():
    configure_gpu()

    (x_train, y_train), (x_test, y_test) = load_data()
    model = build_model()

    train_model(model, x_train, y_train, x_test, y_test)

    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    logger.info(f"Final test accuracy: {accuracy * 100:.2f}%")

    onnx_path = os.path.join("models", "mnist_model.onnx")
    convert_to_onnx(model, onnx_path)

if __name__ == "__main__":
    main()
