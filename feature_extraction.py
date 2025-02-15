import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
import shutil

real_images_path = "Dataset/training_real"
fake_images_path = "Dataset/training_fake"
feature_folder = "extracted_features"

if not os.path.exists(feature_folder):
    os.makedirs(feature_folder)

mobilenet = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
mobilenet.trainable = False  # Freeze the base model

def extract_features(img_path):
    """Extract deep features from MobileNetV2."""
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    
    features = mobilenet.predict(img_array)
    return features.flatten()

def save_features(image_folder, label):
    """Extract features and save in a separate folder."""
    save_path = os.path.join(feature_folder, label)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    for img_name in os.listdir(image_folder):
        img_path = os.path.join(image_folder, img_name)
        if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            features = extract_features(img_path)
            np.save(os.path.join(save_path, img_name.split('.')[0]), features)


save_features(real_images_path, "real")
save_features(fake_images_path, "fake")
