import tkinter as tk
from tkinter import filedialog
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

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Initialize Tkinter GUI root window
root = tk.Tk()
root.title("Flux Stain Detector")

# Initialize folder variables
with_flux_folder = None
without_flux_folder = None
output_folder = None

# Global variable for stopping the training
stop_training = False

# Function to stop training
def stop_training_func():
    global stop_training
    stop_training = True
    logging.info("User has requested to stop training.")

# Function to select folder and update status label
def select_folder(folder_type, status_label):
    folder_selected = filedialog.askdirectory()
    if folder_selected:
        status_label.config(text="Selected", fg="green")
        if folder_type == 'With Flux':
            global with_flux_folder
            with_flux_folder = folder_selected
        elif folder_type == 'Without Flux':
            global without_flux_folder
            without_flux_folder = folder_selected
        elif folder_type == 'Model Output':
            global output_folder
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
    ax1.set_title('Model Accuracy (%)')
    ax1.set(xlabel='epoch', ylabel='accuracy')
    ax2.plot(loss)
    ax2.set_title('Model Loss')
    ax2.set(xlabel='epoch', ylabel='loss')
    plt.show()

# Function to train machine learning model
def train_model():
    global with_flux_folder, without_flux_folder, output_folder, stop_training

    if not with_flux_folder or not without_flux_folder or not output_folder:
        logging.info("Data or output folder not set.")
        return None, None

    with_flux_images, with_flux_labels = load_dataset(with_flux_folder, 1)
    without_flux_images, without_flux_labels = load_dataset(without_flux_folder, 0)
    
    x_data = np.vstack((with_flux_images, without_flux_images))
    y_data = np.hstack((with_flux_labels, without_flux_labels))
    
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)
    
    base_model = ResNet50(weights='imagenet', include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    predictions = Dense(1, activation='sigmoid', kernel_regularizer=l2(0.01))(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    
 # Initialize Adam optimizer with a smaller learning rate
    adam_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

    model.compile(optimizer=adam_optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    
    callbacks = [
        # tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', min_delta=0, patience=3, verbose=0, mode='auto',
             #                            baseline=None, restore_best_weights=True)
    ]
    
    logging.info(f"Training model and saving to {output_folder}")
    history = model.fit(x_train, y_train, epochs=10, batch_size=32, callbacks=callbacks, validation_split=0.2)
    
    model.save(f"{output_folder}/my_model.h5")
    
    return [round(acc * 100, 2) for acc in history.history['accuracy']], history.history['loss']

# New function to plot data in the main thread
def plot_and_train():
    accuracy, loss = train_model()
    if accuracy and loss:
        root.after(0, plot_data, accuracy, loss)  # Schedule plot_data to run in the main thread

# GUI Components
frame = tk.Frame(root)
frame.pack(pady=20)

# Folder selection buttons and status labels
for folder_type in ['With Flux', 'Without Flux', 'Model Output']:
    folder_frame = tk.Frame(frame)
    folder_frame.pack(side="left", padx=10)
    
    status_label = tk.Label(folder_frame, text="Not Selected", fg="red")
    status_label.pack(side="bottom")
    
    button = tk.Button(folder_frame, text=f"Select {folder_type} Folder",
                       command=lambda f=folder_type, s=status_label: select_folder(f, s))
    button.pack()

# Start and stop buttons
start_button = tk.Button(root, text="Start Training", command=lambda: threading.Thread(target=plot_and_train).start())
start_button.pack()

stop_button = tk.Button(root, text="Stop Training", command=stop_training_func)
stop_button.pack()

root.mainloop()
