import tensorflow as tf
import numpy as np
import os
import sys
from tensorflow.keras.preprocessing.image import ImageDataGenerator

MODEL_PATH = 'cancer_detection_model_v3.h5'
DATA_PATH = './data'
IMG_SIZE = (224, 224)
BATCH_SIZE = 1 
OUTPUT_TFLITE = 'cancer_model_quantized.tflite'

print("--- Step 1: Initialization ---")

if not os.path.exists(MODEL_PATH):
    print(f"ERROR: Model file '{MODEL_PATH}' not found.")
    sys.exit()

if not os.path.exists(DATA_PATH):
    print(f"ERROR: Data folder '{DATA_PATH}' not found.")
    sys.exit()

print(f"Loading Keras model: {MODEL_PATH}...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully.")

datagen = ImageDataGenerator(rescale=1./255)

val_gen = datagen.flow_from_directory(
    DATA_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=True
)

if val_gen.samples == 0:
    print("ERROR: No images found in DATA_PATH. Check your folder structure.")
    sys.exit()

print(f"Found {val_gen.samples} images. Using 100 for calibration.")

def representative_data_gen():
    print("Calibration starting...")
    for i in range(100):
        try:
            img, _ = next(val_gen)
            if i % 10 == 0:
                print(f"  Calibrating: Image {i}/100")
            yield [img.astype(np.float32)]
        except StopIteration:
            break
    print("Calibration data feed complete.")

print("--- Step 2: TFLite Conversion ---")
print("This may take a few minutes. Please wait...")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen

converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

try:
    tflite_model = converter.convert()
    print("Conversion logic finished.")
    
    with open(OUTPUT_TFLITE, 'wb') as f:
        f.write(tflite_model)
    print(f"--- SUCCESS ---")
    print(f"Quantized model saved as: {OUTPUT_TFLITE}")
    print(f"File size: {os.path.getsize(OUTPUT_TFLITE) / 1024:.2f} KB")

except Exception as e:
    print(f"--- FAILED ---")
    print(f"An error occurred during conversion: {e}")