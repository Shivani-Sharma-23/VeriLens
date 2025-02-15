import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2, ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, BatchNormalization, AveragePooling2D
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# Paths
real_images_path = "Dataset/training_real"
fake_images_path = "Dataset/training_fake"
feature_folder = "extracted_features"

if not os.path.exists(feature_folder):
    os.makedirs(feature_folder)

def load_features(label):
    """Load extracted features and labels."""
    features, labels = [], []
    label_path = os.path.join(feature_folder, label)
    
    for file in os.listdir(label_path):
        if file.endswith(".npy"):
            feature_vector = np.load(os.path.join(label_path, file))
            features.append(feature_vector)
            labels.append(0 if label == "real" else 1)  # 0 = Real, 1 = Fake
    
    return np.array(features), np.array(labels)

# Load extracted features
X_real, y_real = load_features("real")
X_fake, y_fake = load_features("fake")

# Combine datasets
X = np.vstack((X_real, X_fake))
y = np.hstack((y_real, y_fake))

# ✅ **1. Fully Connected Model**
def create_fc_model(input_shape):
    model = Sequential([
        Dense(512, activation='relu', input_shape=input_shape),
        Dropout(0.5),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

fc_model = create_fc_model((X.shape[1],))

# ✅ **2. ResNet50 Model**
def create_resnet50_model():
    base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False  # Freeze the base model

    x = Flatten()(base_model.output)
    x = Dense(512, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.5)(x)
    output = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=base_model.input, outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.0001), loss="binary_crossentropy", metrics=["accuracy"])
    return model

resnet_model = create_resnet50_model()

# ✅ **3. MesoNet Model**
def create_meso_model(input_shape=(224, 224, 3)):
    model = Sequential([
        Conv2D(8, (3, 3), padding='same', activation='relu', input_shape=input_shape),
        BatchNormalization(),
        AveragePooling2D(pool_size=(2, 2)),

        Conv2D(8, (5, 5), padding='same', activation='relu'),
        BatchNormalization(),
        AveragePooling2D(pool_size=(2, 2)),

        Conv2D(16, (5, 5), padding='same', activation='relu'),
        BatchNormalization(),
        AveragePooling2D(pool_size=(2, 2)),

        Flatten(),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

meso_model = create_meso_model()

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ **Train Fully Connected Model**
print("Training FC Model...")
fc_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)

# ✅ **Train ResNet50 Model**
print("Training ResNet50 Model...")
resnet_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)

# ✅ **Train MesoNet Model**
print("Training MesoNet Model...")
meso_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)

# ✅ **Evaluate Models**
def evaluate_model(model, name):
    loss, acc = model.evaluate(X_val, y_val)
    print(f"{name} - Accuracy: {acc:.4f}, Loss: {loss:.4f}")

evaluate_model(fc_model, "Fully Connected Model")
evaluate_model(resnet_model, "ResNet50 Model")
evaluate_model(meso_model, "MesoNet Model")

# ✅ **Save Models**
model_save_path = "saved_models"
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)

fc_model.save(os.path.join(model_save_path, "fc_model.h5"))
resnet_model.save(os.path.join(model_save_path, "resnet_model.h5"))
meso_model.save(os.path.join(model_save_path, "meso_model.h5"))

print("All models saved successfully!")
