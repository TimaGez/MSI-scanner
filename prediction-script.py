import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

model = tf.keras.models.load_model('cancer_detection_model_v3.h5')

print(model.summary())

# def predict_image(img_path):
#     img = image.load_img(img_path, target_size=(224, 224))
#     img_array = image.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0)
#     img_array /= 255.0                          

#     prediction = model.predict(img_array)
    
#     if prediction[0][0] > 0.5:
#         print(f"Result: POSITIVE ({prediction[0][0]*100:.2f}% confidence)")
#     else:
#         print(f"Result: NEGATIVE ({(1-prediction[0][0])*100:.2f}% confidence)")