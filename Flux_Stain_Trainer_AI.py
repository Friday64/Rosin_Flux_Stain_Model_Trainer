import tkinter as tk
from tkinter import filedialog, ttk
import threading
import logging
import os
import numpy as np
import tensorflow as tf
from keras.applications import MobileNetV2, ResNet50  # You can add other models here
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
selected_model = tk.StringVar(root)
root.title("Flux Stain Detector")

# Initialize folder variables
with_flux_folder = None
without_flux_folder = None
output_folder = None

# Initialize progress bar variable
progress_var = tk.DoubleVar()

# Function to load dataset from folder
def load_dataset(folder_path, label):
    images = []
    labels = []
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        try:
            img = Image.open(img_path).convert('RGB')  # Ensure image is RGB
            img = img.resize((224, 224))  # Resize to the dimensions you want
            img_array = np.array(img)
            images.append(img_array)
            labels.append(label)
        except Exception as e:
            logging.error(f"Error loading image {img_path}: {str(e)}")
    return np.array(images), np.array(labels)

# Function to plot real-time data
def plot_data(accuracy, loss):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(accuracy)
    ax1.set_title('Model Accuracy')
    ax1.set(xlabel='epoch', ylabel='accuracy')
    ax2.plot(loss)
    ax2.set_title('Model Loss')
    ax2.set(xlabel='epoch', ylabel='loss')
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    canvas.get_tk_widget().pack()

# Function to stop training
stop_training = False
def stop_training_func():
    global stop_training
    stop_training = True

def focus_on_flux(image_array):
    mask = (image_array[:,:,0] > 100) & (image_array[:,:,1] > 80) & (image_array[:,:,2] < 100)
    mask = np.stack([mask]*3, axis=-1)
    overlay = np.zeros_like(image_array)
    overlay[mask] = [255, 255, 255]  # using white overlay, you can choose another color
    alpha = 0.3  # transparency level
    img_overlay = (image_array * (1 - alpha) + overlay * alpha).astype(np.uint8)
    return img_overlay

# Function to check if the average color of the image is flux color (yellow-brownish)
def is_flux_color(image_array):
    r, g, b = np.mean(image_array[:,:,0]), np.mean(image_array[:,:,1]), np.mean(image_array[:,:,2])
    return 100 < r < 200 and 50 < g < 150 and 0 < b < 100  # Adjust these ranges based on the actual color of the flux

# Function to train machine learning model
def train_model():
    global with_flux_folder, without_flux_folder, output_folder
    if not with_flux_folder or not without_flux_folder or not output_folder:
        logging.info("Data or output folder not set.")
        return

    with_flux_images, with_flux_labels = load_dataset(with_flux_folder, 1)
    without_flux_images, without_flux_labels = load_dataset(without_flux_folder, 0)

    # Filter images based on color
    with_flux_images = [img for img in with_flux_images if is_flux_color(img)]
    without_flux_images = [img for img in without_flux_images if not is_flux_color(img)]
    
    x_data = np.vstack((with_flux_images, without_flux_images))
    y_data = np.hstack((with_flux_labels[:len(with_flux_images)], without_flux_labels[:len(without_flux_images)]))
    
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)
    
    base_model = MobileNetV2(weights='imagenet', include_top=False)  # Initialize MobileNetV2
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    logging.info(f"Training model and saving to {output_folder}")
    history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))
    model.save(f"{output_folder}/my_model.h5")
    
    plot_data(history.history['accuracy'], history.history['loss'])


# Function to select folder and update corresponding global variable and status label
def select_folder(var, status_label):
    folder = filedialog.askdirectory(title=f"Select {var} Folder")
    global with_flux_folder, without_flux_folder, output_folder
    if folder:
        if var == 'With Flux':
            with_flux_folder = folder
        elif var == 'Without Flux':
            without_flux_folder = folder
        elif var == 'Model Output':
            output_folder = folder
        logging.info(f"{var} folder selected: {folder}")
        status_label.config(text="Selected", fg="green")

# GUI Components
# Create status labels for folders
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

progress_bar = ttk.Progressbar(root, variable=progress_var, maximum=100)
progress_bar.pack()

root.mainloop()
