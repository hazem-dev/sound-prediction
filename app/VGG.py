# Import necessary libraries
import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image

# Set the path to your dataset
dataset_path = "./static/images_original"

# Define VGG model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze convolutional layers
for layer in base_model.layers:
    layer.trainable = False

# Build a custom model on top of VGG
model = models.Sequential()
model.add(base_model)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(len(os.listdir(dataset_path)), activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# Function to preprocess the image
def preprocess_image(img_path):
    print(f"Processing image: {img_path}")
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array


# Function to load and preprocess the dataset
def load_dataset(dataset_path):
    data = []
    labels = []
    class_indices = {label: i for i, label in enumerate(os.listdir(dataset_path))}

    for label in os.listdir(dataset_path):
        label_path = os.path.join(dataset_path, label)
        for img in os.listdir(label_path):
            img_path = os.path.join(label_path, img)
            img_array = preprocess_image(img_path)
            data.append(img_array)
            labels.append(class_indices[label])

    return np.vstack(data), np.array(labels)


# Load and preprocess the dataset
print("Loading and preprocessing dataset...")
x_train, y_train = load_dataset(dataset_path)

# Convert labels to one-hot encoding
y_train = tf.keras.utils.to_categorical(y_train)

# Train the model
print("Training the model...")
model.fit(x_train, y_train, epochs=5)  # You may need to adjust the number of epochs

# Save the model to a pickle file
model_pickle_path = "/app/models/vgg_model.pkl"
print(f"Saving the model to {model_pickle_path}...")
with open(model_pickle_path, 'wb') as model_pkl:
    pickle.dump(model, model_pkl)
