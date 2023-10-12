import tkinter as tk
from tkinter import filedialog, ttk
import threading
import logging
import os
import numpy as np
import tensorflow as tf
from keras.applications import ResNet50
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.regularizers import l2
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

# Initialize a variable to control training
stop_training = False

def stop_training_func():
    global stop_training
    stop_training = True
    logging.info("User has requested to stop training.")

# Function to select folder and update status label
def select_folder(folder_type, status_label):
    global with_flux_folder, without_flux_folder, output_folder
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
        try:
            img = Image.open(img_path).convert('RGB')
            img = img.resize((224, 224))
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
    ax1.set_title('Model Accuracy (%)')  # Display as %
    ax1.set(xlabel='epoch', ylabel='accuracy (%)')
    ax2.plot(loss)
    ax2.set_title('Model Loss')
    ax2.set(xlabel='epoch', ylabel='loss')
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    canvas.get_tk_widget().pack()

# Function to train machine learning model
def train_model():
    global with_flux_folder, without_flux_folder, output_folder, stop_training

    if not with_flux_folder or not without_flux_folder or not output_folder:
        logging.info("Data or output folder not set.")
        return

    with_flux_images, with_flux_labels = load_dataset(with_flux_folder, 1)
    without_flux_images, without_flux_labels = load_dataset(without_flux_folder, 0)
    
    x_data = np.vstack((with_flux_images, without_flux_images))
    y_data = np.hstack((with_flux_labels, without_flux_labels))
    
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)
    
    base_model = ResNet50(weights='imagenet', include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)  # Added Dropout layer
    predictions = Dense(1, activation='sigmoid', kernel_regularizer=l2(0.01))(x)  # Added L2 regularization
    model = Model(inputs=base_model.input, outputs=predictions)
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', min_delta=0, patience=3, verbose=0, mode='auto',
                                         baseline=None, restore_best_weights=True)
    ]
    
    logging.info(f"Training model and saving to {output_folder}")
    history = model.fit(x_train, y_train, epochs=10, batch_size=32, callbacks=callbacks, validation_split=0.2)
    
    model.save(f"{output_folder}/my_model.h5")
    
    plot_data(history.history['accuracy'], history.history['loss'])

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

# Added Stop button
stop_button = tk.Button(root, text="Stop Training", command=stop_training_func)
stop_button.pack()

root.mainloop()
