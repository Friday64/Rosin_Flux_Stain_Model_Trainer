# Standard library imports
import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping, Callback
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import tkinter as tk
from tkinter import messagebox
import threading
import tensorrt as trt

# Paths to image folders and model
with_flux_folder = "/home/matt/desktop/With_Flux"
without_flux_folder = "/home/matt/desktop/Without_Flux"
output_folder = "/home/matt/desktop/Flux_Models"
trt_engine_path = "/home/matt/desktop/Flux_Models/flux_model.trt"

# Size to which images will be resized
img_size = (128, 128)

# Global flag to control training
stop_training = False

# Custom callback to stop training
class CustomStopTrainingCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        global stop_training
        if stop_training:
            print("Stopping training...")
            self.model.stop_training = True

# Function to preprocess and load images
def preprocess_and_load_images(directory_path, img_size):
    # [Your existing code for preprocessing and loading images]

# Function to create the machine learning model
def create_model(input_shape=(128, 128, 1), num_classes=2):
    # [Your existing code for creating the model]

# Function to train the model in a separate thread
def train_model_thread(epochs, model, callbacks_list):
    # [Your existing code for training the model]

# Function to start training in a separate thread
def start_training_thread():
    global window
    try:
        epochs = int(epochs_entry.get())
        model_file_path = f"{output_folder}/flux_model.h5"
        if os.path.exists(model_file_path):
            model = load_model(model_file_path)
        else:
            model = create_model(input_shape=(128, 128, 1), num_classes=2)
        callbacks_list = [EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True), CustomStopTrainingCallback()]
        training_thread = threading.Thread(target=train_model_thread, args=(epochs, model, callbacks_list))
        training_thread.start()
    except ValueError:
        messagebox.showerror("Error", "Number of epochs is not a valid integer.")

# Function to stop training
def halt_training():
    global stop_training
    stop_training = True

# TensorRT Inference class
class TRTInference:
    def __init__(self, engine_path):
        self.engine = self.load_engine(engine_path)
        self.context = self.engine.create_execution_context()

    @staticmethod
    def load_engine(engine_path):
        with open(engine_path, 'rb') as f, trt.Runtime(trt.Logger(trt.Logger.WARNING)) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    def infer(self, input_data):
        # [Your existing code for inference using TensorRT]

# Initialize the TensorRT inference object
trt_inference = TRTInference(trt_engine_path)

# Initialize Tkinter window
window = tk.Tk()
window.title("Flux Stain Detector")

# Create UI elements
tk.Label(window, text="Number of Epochs:").pack()
epochs_entry = tk.Entry(window)
epochs_entry.pack()

train_button = tk.Button(window, text="Train Model", command=start_training_thread)
train_button.pack()

stop_button = tk.Button(window, text="Stop Training", command=halt_training)
stop_button.pack()

# Run the Tkinter event loop
window.mainloop()
