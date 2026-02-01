# -*- coding: utf-8 -*-

import os
import tensorflow as tf
import matplotlib.pyplot as plt
import time

start_time = time.time()  # record start

print("TensorFlow version:", tf.version)
print("GPUs:", tf.config.list_physical_devices("GPU"))

if tf.config.list_physical_devices("GPU"):
    print("GPU is visible to TensorFlow.")
else:
    print("No GPU detected by TensorFlow.")

# ---------------- CONFIG ----------------

DATA_DIR = r"C:\Users\jblair.spyder-py3\birds_dataset2\birds_dataset"

IMG_SIZE = 300
BATCH_SIZE = 32  # 32
AUTOTUNE = tf.data.AUTOTUNE

# ---------------- DATA ----------------

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    labels="inferred",
    label_mode="categorical",
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    labels="inferred",
    label_mode="categorical",
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
)

class_names = train_ds.class_names
num_classes = len(class_names)
print("Classes:", class_names)

# Simple on-the-fly augmentation (TRAINING ONLY)

data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.05),
        tf.keras.layers.RandomZoom(0.1),
    ]
)

# Cast to float32, optional augment, cache and prefetch


def preprocess_ds(ds, augment=False):
    # cast
    ds = ds.map(
        lambda x, y: (tf.cast(x, tf.float32), y),
        num_parallel_calls=AUTOTUNE,
    )

    # apply augmentation only when requested (training)
    if augment:
        ds = ds.map(
            lambda x, y: (data_augmentation(x, training=True), y),
            num_parallel_calls=AUTOTUNE,
        )

    return ds.cache().prefetch(AUTOTUNE)


# Augment only training set

train_ds = preprocess_ds(train_ds.shuffle(1000), augment=True)
val_ds = preprocess_ds(val_ds, augment=False)

# ---------------- MODEL ----------------

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights="imagenet",
)
base_model.trainable = False  # freeze base for initial training

inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.3)(x)  # 20% dropout rate
outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

# ---------------- TRAIN (head only) ----------------

history = model.fit(
    train_ds,
    epochs=9,
    validation_data=val_ds,
)

# ---------------- FINE-TUNING ----------------

# Unfreeze last part of the base model for a few more epochs

base_model.trainable = True
fine_tune_at = len(base_model.layers) - 30  # last ~30 layers

for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

history_ft = model.fit(
    train_ds,
    epochs=4,
    validation_data=val_ds,
)

# ---------------- SAVE KERAS MODEL ----------------

saved_model_dir = r"C:\Users\jblair.spyder-py3\birds_dataset2\bird_classifier_saved_model3"

# Make sure parent directory exists
os.makedirs(os.path.dirname(saved_model_dir), exist_ok=True)

# Export a TF SavedModel (Keras 3+ way)
model.export(saved_model_dir)
print("Exported TF SavedModel to:", saved_model_dir)

# ---------------- CONVERT TO QUANTIZED TFLITE ----------------

converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

tflite_path = r"C:\Users\jblair.spyder-py3\birds_dataset2\bird_classifier_int8.tflite"
with open(tflite_path, "wb") as f:
    f.write(tflite_model)

print("Saved quantized TFLite model to:", tflite_path)
print("Classes:", class_names)

# ---------------- PLOTTING ----------------

PLOT_DIR = r"C:\Users\jblair.spyder-py3\birds_dataset2\training_plots"


def plot_history(history, history_ft=None, out_dir=PLOT_DIR):
    os.makedirs(out_dir, exist_ok=True)

    # Collect metrics from both phases (initial + fine-tune)
    acc = history.history.get("accuracy", [])
    val_acc = history.history.get("val_accuracy", [])
    loss = history.history.get("loss", [])
    val_loss = history.history.get("val_loss", [])

    if history_ft is not None:
        acc += history_ft.history.get("accuracy", [])
        val_acc += history_ft.history.get("val_accuracy", [])
        loss += history_ft.history.get("loss", [])
        val_loss += history_ft.history.get("val_loss", [])

    epochs = range(1, len(acc) + 1)

    # Accuracy plot
    plt.figure()
    plt.plot(epochs, acc, label="Training accuracy")
    plt.plot(epochs, val_acc, label="Validation accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and validation accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    acc_path = os.path.join(out_dir, "accuracy.png")
    plt.savefig(acc_path, dpi=150)
    print("Saved accuracy plot to:", acc_path)
    plt.show()  # show on screen

    # Loss plot
    plt.figure()
    plt.plot(epochs, loss, label="Training loss")
    plt.plot(epochs, val_loss, label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and validation loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    loss_path = os.path.join(out_dir, "loss.png")
    plt.savefig(loss_path, dpi=150)
    print("Saved loss plot to:", loss_path)
    plt.show()  # show on screen


# Call this right after training & fine-tuning:
plot_history(history, history_ft)

# ---------------- EXPORT SAVEDMODEL + TFLITE ----------------

saved_model_dir = r"C:\Users\jblair.spyder-py3\birds_dataset2\bird_classifier_saved_model3"
os.makedirs(os.path.dirname(saved_model_dir), exist_ok=True)

print("\nExporting SavedModel to:", saved_model_dir)
model.export(saved_model_dir)
print("Export complete.")

# Convert to quantized TFLite

print("\nConverting to quantized TFLite...")
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
print("TFLite conversion complete.")

tflite_path = r"C:\Users\jblair.spyder-py3\birds_dataset2\bird_classifier_int8.tflite"
print("Writing TFLite file to:", tflite_path)
with open(tflite_path, "wb") as f:
    f.write(tflite_model)

if os.path.exists(tflite_path):
    size_bytes = os.path.getsize(tflite_path)
    print(f"Saved quantized TFLite model to: {tflite_path} (size: {size_bytes} bytes)")
else:
    print(" ERROR: TFLite file was not created.")

print("Classes:", class_names)

end_time = time.time()  # record end
elapsed_time = (end_time - start_time) / 60
print(f"Script took {elapsed_time:.2f} minutes")
