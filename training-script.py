import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import os

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20 
DATA_PATH = './data' 

datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

train_gen = datagen.flow_from_directory(
    DATA_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training',
    shuffle=True
)

val_gen = datagen.flow_from_directory(
    DATA_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation',
    shuffle=False 
)

print(f"Detected Classes: {train_gen.class_indices}")

base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False 

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    layers.Dropout(0.4), 
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=optimizers.Adam(learning_rate=0.00001),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.Recall(name='recall'), tf.keras.metrics.Precision(name='precision')]
)

print("Starting fresh training...")
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS
)

model.save('cancer_detection_model_v3.h5')

def save_performance_graphs(history):
    metrics = ['accuracy', 'loss', 'recall', 'precision']
    plt.figure(figsize=(16, 12))
    
    for i, metric in enumerate(metrics):
        plt.subplot(2, 2, i+1)
        plt.plot(history.history[metric], label=f'Train {metric}', color='blue', lw=2)
        plt.plot(history.history[f'val_{metric}'], label=f'Val {metric}', color='orange', lw=2)
        plt.title(f'Training vs Validation {metric.capitalize()}')
        plt.xlabel('Epochs')
        plt.ylabel(metric.capitalize())
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig('full_training_report.png')
    plt.show()
    print("Graphs saved as 'full_training_report.png'")

save_performance_graphs(history)

print("Generating Confusion Matrix...")
val_gen.reset()
predictions = model.predict(val_gen)
y_pred = (predictions > 0.5).astype(int)
y_true = val_gen.classes

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
            xticklabels=list(train_gen.class_indices.keys()), 
            yticklabels=list(train_gen.class_indices.keys()))
plt.title('Final Confusion Matrix')
plt.ylabel('Actual Truth')
plt.xlabel('Model Prediction')
plt.savefig('confusion_matrix_v3.png')
plt.show()
print("Confusion Matrix saved as 'confusion_matrix_v3.png'")