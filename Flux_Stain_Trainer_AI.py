from PIL import Image  # Changed from tkinter Image
from keras.preprocessing.image import ImageDataGenerator
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import gradio as gr
import numpy as np
import os

# Define your data directory
data_dir = "./data"

# Initialize ImageDataGenerator
datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# Create a training image generator
train_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

# Model definition
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

# Model compilation
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

# Function to preprocess image
def preprocess_image(image):
    image = Image.fromarray(image.astype('uint8'), 'RGB')
    image = image.resize((150, 150))
    image_array = np.array(image)
    image_array = image_array / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

# Initialize empty lists to hold images and labels
collected_images = []
collected_labels = []

# Function to train on batch
def train_on_batch(images, labels):
    history = model.fit(np.vstack(images), labels, epochs=10)
    print(history.history)  # Just printing metrics for now

# Gradio UI function
def classify_image(image, choice):
    global collected_images, collected_labels

    # Preprocess image and label
    preprocessed_image = preprocess_image(image)
    label = 1 if choice == 'With Flux' else 0
    
    # Collect for batch
    collected_images.append(preprocessed_image)
    collected_labels.append(label)

    # Train in batches
    if len(collected_images) >= 32:
        train_on_batch(np.array(collected_images), np.array(collected_labels))
        collected_images, collected_labels = [], []

    return 'Processed'

# Gradio Interface
iface = gr.Interface(
    fn=classify_image, 
    inputs=["image", gr.inputs.Radio(["With Flux", "Without Flux"])], 
    outputs="text"
)

iface.launch()
