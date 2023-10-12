import tkinter as tk
from tkinter import filedialog
import threading
import logging
import os
import numpy as np
import tensorflow as tf
from keras.applications import ResNet50
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Initialize Tkinter GUI root window
root = tk.Tk()
root.title("Flux Stain Detector")

# Initialize folder variables
with_flux_folder = None
without_flux_folder = None
output_folder = None

# Initialize the stop_training flag to control the training loop
stop_training = False

# Function to stop training
def stop_training_func():
    global stop_training
    stop_training = True
    logging.info("User has requested to stop training.")

# Function to select folder and update status label
def select_folder(folder_type, status_label):
    global with_flux_folder, without_flux_folder, output_folder  # Declare them as global
    folder_selected = filedialog.askdirectory()
    if folder_selected:
        status_label.config(text="Selected", fg="green")
        if folder_type == 'With Flux':
            with_flux_folder = folder_selected
        elif folder_type == 'Without Flux':
            without_flux_folder = folder_selected
        elif folder_type == 'Model Output':
            output_folder = folder_selected

# Function to load dataset from folder
def load_dataset(folder_path, label):
    images = []
    labels = []
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        img = Image.open(img_path).convert('RGB')
        img = img.resize((224, 224))
        img_array = np.array(img)
        images.append(img_array)
        labels.append(label)
    return np.array(images), np.array(labels)

# Function to plot real-time data
def plot_data(accuracy, loss):
    # Multiply accuracy by 100 to convert to percentage
    accuracy_percentage = [acc * 100 for acc in accuracy]
    
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(accuracy_percentage)  # Use accuracy_percentage here
    ax1.set_title('Model Accuracy (%)')  # Updated title to indicate percentage
    ax1.set(xlabel='epoch', ylabel='accuracy (%)')  # Updated ylabel to indicate percentage
    ax2.plot(loss)
    ax2.set_title('Model Loss')
    ax2.set(xlabel='epoch', ylabel='loss')
    
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    canvas.get_tk_widget().pack()


# Function to train machine learning model
def train_model():
    global stop_training  # To control the loop
    
    if not with_flux_folder or not without_flux_folder or not output_folder:
        logging.info("Data or output folder not set.")
        return

    with_flux_images, with_flux_labels = load_dataset(with_flux_folder, 1)
    without_flux_images, without_flux_labels = load_dataset(without_flux_folder, 0)

    x_data = np.vstack((with_flux_images, without_flux_images))
    y_data = np.hstack((with_flux_labels, without_flux_labels))

    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)

    # Using ResNet50 for better feature extraction
    base_model = ResNet50(weights='imagenet', include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    logging.info(f"Training model and saving to {output_folder}")
    history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
    
    # Check if the stop_training flag is set, if so stop training and exit
    if stop_training:
        logging.info("Stopping training as per user request.")
        return

    model.save(f"{output_folder}/my_model.h5")

# GUI components
with_flux_status_label = tk.Label(root, text="Not Selected", fg="red")
with_flux_status_label.pack()

select_with_flux_folder_button = tk.Button(root, text="Select With Flux Folder", 
                                           command=lambda: select_folder('With Flux', with_flux_status_label))
select_with_flux_folder_button.pack()

without_flux_status_label = tk.Label(root, text="Not Selected", fg="red")
without_flux_status_label.pack()

select_without_flux_folder_button = tk.Button(root, text="Select Without Flux Folder", 
                                              command=lambda: select_folder('Without Flux', without_flux_status_label))
select_without_flux_folder_button.pack()

output_folder_status_label = tk.Label(root, text="Not Selected", fg="red")
output_folder_status_label.pack()

select_output_folder_button = tk.Button(root, text="Select Model Output Folder", 
                                        command=lambda: select_folder('Model Output', output_folder_status_label))
select_output_folder_button.pack()

start_button = tk.Button(root, text="Start Training", command=train_model)
start_button.pack()

stop_button = tk.Button(root, text="Stop Training", command=stop_training_func)
stop_button.pack()

root.mainloop()
