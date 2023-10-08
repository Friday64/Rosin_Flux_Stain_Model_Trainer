import tkinter as tk
from tkinter import filedialog, PhotoImage
from PIL import Image, ImageTk
import threading
import logging
import os
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import load_img, img_to_array
from keras.applications import MobileNetV2
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from sklearn.model_selection import train_test_split

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Initialize Tkinter GUI root window
root = tk.Tk()
root.title("Flux Stain Detector")

# Initialize data_folder and output_folder variables
with_flux_folder = None
without_flux_folder = None
output_folder = None

# Initialize image index
image_index = 0

# Function to load dataset from folder
def load_dataset(folder_path, label):
    images = []
    labels = []
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        img = load_img(img_path, target_size=(224, 224))
        img_array = img_to_array(img)
        images.append(img_array)
        labels.append(label)
    return np.array(images), np.array(labels)

# Function to display images from a folder
def display_images(folder, index):
    if folder:
        image_files = os.listdir(folder)
        img_path = os.path.join(folder, image_files[index])
        img = Image.open(img_path)
        img = img.resize((200, 200), Image.ANTIALIAS)
        img = ImageTk.PhotoImage(img)
        img_label.config(image=img)
        img_label.image = img

# Function to navigate images
def next_image(folder):
    global image_index
    image_index += 1
    image_index %= len(os.listdir(folder))
    display_images(folder, image_index)

# Function to train the machine learning model
def train_model():
    global with_flux_folder, without_flux_folder, output_folder
    try:
        if not with_flux_folder or not without_flux_folder or not output_folder:
            logging.info("Data or output folder not set.")
            return
        with_flux_images, with_flux_labels = load_dataset(with_flux_folder, 1)
        without_flux_images, without_flux_labels = load_dataset(without_flux_folder, 0)

        x_data = np.vstack((with_flux_images, without_flux_images))
        y_data = np.hstack((with_flux_labels, without_flux_labels))

        x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)

        base_model = MobileNetV2(weights='imagenet', include_top=False)
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        predictions = Dense(1, activation='sigmoid')(x)
        model = Model(inputs=base_model.input, outputs=predictions)

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        logging.info(f"Training model and saving to {output_folder}")

        model.fit(x_train, y_train, epochs=5, batch_size=32)
        model.save(f"{output_folder}/my_model.h5")

    except Exception as e:
        logging.error(f"An error occurred during training: {e}")

# Function to open folder dialog for with_flux folder
def select_with_flux_folder():
    global with_flux_folder
    with_flux_folder = filedialog.askdirectory(title="Select With Flux Folder")
    if with_flux_folder:
        logging.info(f"With Flux folder selected: {with_flux_folder}")
        display_images(with_flux_folder, 0)

# Function to open folder dialog for without_flux folder
def select_without_flux_folder():
    global without_flux_folder
    without_flux_folder = filedialog.askdirectory(title="Select Without Flux Folder")
    if without_flux_folder:
        logging.info(f"Without Flux folder selected: {without_flux_folder}")
        display_images(without_flux_folder, 0)

# Function to open folder dialog for model output folder
def select_output_folder():
    global output_folder
    output_folder = filedialog.askdirectory(title="Select Model Output Folder")
    if output_folder:
        logging.info(f"Output folder selected: {output_folder}")

# Function to start the training thread
def start_training_thread():
    train_thread = threading.Thread(target=train_model)
    train_thread.start()

# GUI Components
img_label = tk.Label(root)
img_label.pack()

select_with_flux_folder_button = tk.Button(root, text="Select With Flux Folder", command=select_with_flux_folder)
select_with_flux_folder_button.pack()

next_with_flux_image_button = tk.Button(root, text="Next With Flux Image", command=lambda: next_image(with_flux_folder))
next_with_flux_image_button.pack()

select_without_flux_folder_button = tk.Button(root, text="Select Without Flux Folder", command=select_without_flux_folder)
select_without_flux_folder_button.pack()

next_without_flux_image_button = tk.Button(root, text="Next Without Flux Image", command=lambda: next_image(without_flux_folder))
next_without_flux_image_button.pack()

select_output_folder_button = tk.Button(root, text="Select Model Output Folder", command=select_output_folder)
select_output_folder_button.pack()

start_button = tk.Button(root, text="Start Training", command=start_training_thread)
start_button.pack()

status_label = tk.Label(root, text="Status: Awaiting command")
status_label.pack()

root.mainloop()
