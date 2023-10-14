# Import required libraries
from PIL import Image
import gradio as gr
import numpy as np
import os
import matplotlib.pyplot as plt
import shutil
import logging
import time
from keras.preprocessing.image import ImageDataGenerator
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.callbacks import TensorBoard

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Initialize TensorBoard
tensorboard = TensorBoard(log_dir=f"./logs/{time.time()}")

# Define the data directory path (modify this)
data_dir = "C:/Users/Matthew/Desktop/Flux_Stain_Project_Pics"

# Define ImageDataGenerator
datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# Define model architecture
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

# Initialize empty lists to hold images and labels
collected_images = []
collected_labels = []

# Additional function to preprocess image
def preprocess_image(image):
    image = Image.fromarray(image.astype('uint8'), 'RGB')
    image = image.resize((150, 150))
    image_array = np.array(image) / 255.0
    return np.expand_dims(image_array, axis=0)

# Additional function to train model on batch
def train_on_batch(images, labels):
    images = np.vstack(images)
    labels = np.array(labels)
    start_time = time.time()
    history = model.train_on_batch(images, labels, callbacks=[tensorboard])
    elapsed_time = time.time() - start_time
    logging.info(f"Training took {elapsed_time:.2f} seconds")
    logging.info(f"Training metrics: {history}")

# Gradio UI function
def classify_image(image, choice):
    global collected_images, collected_labels
    try:
        preprocessed_image = preprocess_image(image)
        label = 1 if choice == 'With Flux' else 0
        collected_images.append(preprocessed_image)
        collected_labels.append(label)
        if len(collected_images) >= 32:
            train_on_batch(np.array(collected_images), np.array(collected_labels))
            collected_images, collected_labels = [], []
        return f"Image processed and labeled as {'With Flux' if label == 1 else 'Without Flux'}"
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        return f"An error occurred: {e}"
# Function to save the model
def save_model(model, model_name="my_model.h5"):
    folder_path = os.path.join(os.path.expanduser("~"), 'Desktop', 'Flux_Models')
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    model_path = os.path.join(folder_path, model_name)
    model.save(model_path)
    logging.info(f"Model saved to {model_path}")

# Gradio Interface
iface = gr.Interface(
    fn=classify_image,
    inputs=["image", gr.inputs.Radio(["With Flux", "Without Flux"])],
    outputs="text"
)

iface.launch()
