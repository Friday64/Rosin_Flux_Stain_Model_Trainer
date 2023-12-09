# Standard library imports
import os  # For operating system dependent functionality
import numpy as np  # NumPy for numerical operations
import cv2  # OpenCV for computer vision tasks
import tensorflow as tf  # TensorFlow for machine learning and neural network operations
from keras.models import Sequential, load_model # Sequential and load_model for neural network operations
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D  # Various layers for neural networks
from keras.callbacks import EarlyStopping, Callback  # Callbacks for training control
from keras.utils import to_categorical  # to_categorical for label encoding
from keras.optimizers import Adam  # Adam optimizer for training
from sklearn.model_selection import train_test_split  # train_test_split to split data
import tkinter as tk  # tkinter for GUI applications
import threading # threading for running operations in parallel

# TensorRT and CUDA imports for inference on Jetson Nano
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

# Paths to image folders and model
with_flux_folder = "/home/matt/desktop/With_Flux"
without_flux_folder = "/home/matt/desktop/Without_Flux"
output_folder = "/home/matt/desktop/Flux_Models"
trt_engine_path = "/home/matt/desktop/Flux_Models/flux_model.trt"  # Path to the TensorRT engine file

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
    image_files = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    dataset = np.zeros((len(image_files), img_size[0], img_size[1]), dtype=np.float32)
    for idx, image_file in enumerate(image_files):
        try:
            img = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, img_size)
                img = img / 255.0
                dataset[idx] = img
            else:
                print(f"Warning: Image {image_file} could not be loaded and will be skipped.")
        except Exception as e:
            print(f"Error processing image {image_file}: {e}")
    print(f"Finished loading and preprocessing {len(dataset)} images from {directory_path}")
    return dataset

# Function to create the machine learning model
def create_model(input_shape=(128, 128, 1), num_classes=2):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer=Adam(learning_rate=0.00001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Function to train the model in a separate thread
def train_model_thread(epochs, model, callbacks_list):
    callbacks_list.append(CustomStopTrainingCallback())
    train_data = preprocess_and_load_images(with_flux_folder, img_size)
    train_labels = np.ones(train_data.shape[0])
    test_data = preprocess_and_load_images(without_flux_folder, img_size)
    test_labels = np.zeros(test_data.shape[0])
    data = np.vstack([train_data, test_data])
    labels = np.hstack([train_labels, test_labels])
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    try:
        history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs, batch_size=64, callbacks=callbacks_list, verbose=1)
        if not stop_training:
            model.save(f"{output_folder}/flux_model.h5")
            print("Model trained and saved successfully.")
        else:
            print("Training was stopped prematurely.")
    except Exception as e:
        print(f"An error occurred during training: {e}")

# Function to start training in a separate thread
def start_training_thread():
    try:
        epochs = int(epochs_entry.get())
        model_file_path = f"{output_folder}/flux_model.h5"
        if os.path.exists(model_file_path):
            print("Loading existing model for retraining...")
            model = load_model(model_file_path)
        else:
            print("Creating a new model...")
            model = create_model(input_shape=(128, 128, 1), num_classes=2)
        callbacks_list = [EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)]
        training_thread = threading.Thread(target=train_model_thread, args=(epochs, model, callbacks_list))
        training_thread.start()
    except ValueError:
        print("Number of epochs is not a valid integer.")

# Function to stop training
def halt_training():
    global stop_training
    stop_training = True
    print("Training stopped by user.")

# TensorRT Inference class
class TRTInference:
    def __init__(self, engine_path):
        self.engine = self.load_engine(engine_path)
        self.context = self.engine.create_execution_context()

    @staticmethod
    def load_engine(engine_path):
        with open(engine_path, 'rb') as f, trt.Runtime(trt.Logger(trt.Logger.WARNING)) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    def allocate_buffers(self):
        h_input = cuda.pagelocked_empty(trt.volume(self.engine.get_binding_shape(0)), dtype=np.float32)
        h_output = cuda.pagelocked_empty(trt.volume(self.engine.get_binding_shape(1)), dtype=np.float32)
        d_input = cuda.mem_alloc(h_input.nbytes)
        d_output = cuda.mem_alloc(h_output.nbytes)
        return h_input, d_input, h_output, d_output

    def infer(self, input_data):
        h_input, d_input, h_output, d_output = self.allocate_buffers()
        cuda.memcpy_htod(d_input, input_data)
        self.context.execute_v2(bindings=[int(d_input), int(d_output)])
        cuda.memcpy_dtoh(h_output, d_output)
        return h_output

# Initialize the TensorRT inference object
trt_inference = TRTInference(trt_engine_path)

# Initialize Tkinter window
window = tk.Tk()
window.title("Flux Stain Detector")

# Create UI elements
tk.Label(window, text="Number of Epochs:").pack()
epochs_entry = tk.Entry(window)
epochs_entry.pack()

# Modify the "Start Training" button to use start_training_thread
train_button = tk.Button(window, text="Train Model", command=start_training_thread)
train_button.pack()

# Modify the "Stop Training" button to use halt_training
stop_button = tk.Button(window, text="Stop Training", command=halt_training)
stop_button.pack()

# Run the Tkinter event loop
window.mainloop()
