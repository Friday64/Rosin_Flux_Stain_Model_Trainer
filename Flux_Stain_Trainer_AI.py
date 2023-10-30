import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from keras.regularizers import l2, l1
import tensorflow_model_optimization as tfmot
from keras.callbacks import LambdaCallback
import tensorflow as tf
import threading

# Global Variables
with_flux_folder = ""
without_flux_folder = ""
output_folder = "output"
stop_training = False

def browse_folder(label_widget, folder_type):
    folder = filedialog.askdirectory(title=f"Select {folder_type} Folder")
    if folder:
        label_widget.config(text=folder)
        global with_flux_folder, without_flux_folder
        if folder_type == "With Flux":
            with_flux_folder = folder
        else:
            without_flux_folder = folder

def browse_output(label_widget):
    global output_folder
    output_folder = filedialog.askdirectory(title="Select Output Folder")
    if output_folder:
        label_widget.config(text=output_folder)
def preprocess_and_load_images(directory_path, img_size):
    """Load and preprocess images from a directory."""
    images = []
    labels = []

    # Determine label based on folder name
    folder_name = os.path.basename(directory_path)
    if folder_name == "WithFlux":  # Change "WithFlux" to the exact folder name you're using
        label = 1
    else:
        label = 0

    # Iterate through each image in the directory
    for img_path in os.listdir(directory_path):
        # Read the image
        img = cv2.imread(os.path.join(directory_path, img_path))
        
        # Check if the image was read properly
        if img is None:
            print(f"Error reading image: {img_path}")
            continue

        # Ensure the image dimensions are not zero
        if img.shape[0] == 0 or img.shape[1] == 0:
            print(f"Image has invalid dimensions: {img_path}")
            continue
        
        # Convert the image to RGB (OpenCV loads images in BGR by default)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Enhance contrast and brightness
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        img = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
        
        # Resize the image to the specified dimensions
        img = cv2.resize(img, (img_size, img_size))
        
        # Normalize the image pixels to be between 0 and 1
        img = img / 255.0
        
        # Append the image and its corresponding label to the lists
        images.append(img)
        labels.append(label)
    
    # Convert lists to numpy arrays
    images = np.array(images)
    labels = np.array(labels)
    
    return list(zip(images, labels))

def create_model():
    """
    Creates a simplified Convolutional Neural Network (CNN) model for classifying images with or without flux stains.

    Returns:
        keras.models.Sequential: The CNN model
    """
    # Initialize the model
    model = keras.Sequential()
      # Modify the input shape to 128x128x3
    model.add(keras.layers.InputLayer(input_shape=(128, 128, 3)))

    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D((2, 2)))
    
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dense(2, activation='softmax'))
    # Add this where you're defining your model architecture
    model.add(Dense(2, activation='softmax'))  # Final output layer


   # Compile the model
    # Update your model compilation line
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    loss='categorical_crossentropy',
    metrics=['accuracy']

    return model

def apply_pruning_to_layers(model):
    """
    Apply pruning to the layers of the model.

    Parameters:
        model (keras.models.Sequential): Original Keras model.

    Returns:
        keras.models.Sequential: Pruned Keras model.
    """
    # Only prune layers that are Conv2D or Dense
    pruning_params = {'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
                        initial_sparsity=0.50, final_sparsity=0.90,
                        begin_step=0, end_step=10000)}

    model_for_pruning = tfmot.sparsity.keras.prune_low_magnitude(model, **pruning_params)

    return model_for_pruning
def convert_to_tflite(model):

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open(os.path.join(output_folder, "Flux_Stain_Model.tflite"), "wb") as f:
        f.write(tflite_model)
    print("Model converted to TFLite format.")

def train_model(epochs, Flux_Model):
  
    # Path to save or load the model
    model_path = os.path.join(output_folder, f'{Flux_Model}.h5')
    
    # Check if the model already exists, if so load it, otherwise create a new one
    if os.path.exists(model_path):
        model = load_model(model_path)
    else:
        model = create_model()
        
    # Apply pruning to the model
    model_for_pruning = apply_pruning_to_layers(model)
    
    # Preprocess and load images
    with_flux_data = preprocess_and_load_images(with_flux_folder, 128)
    without_flux_data = preprocess_and_load_images(without_flux_folder, 128)
    all_data = with_flux_data + without_flux_data
    
    # Prepare the datasets
    X = np.array([i[0] for i in all_data])
    y = to_categorical([i[1] for i in all_data])
    
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    y_train_onehot = to_categorical(y_train)
    y_test_onehot = to_categorical(y_test)

# Call the training function
    threading.Thread(target=train_model, args=(model, X_train, y_train_onehot, X_test, y_test_onehot)).start()  # Update the arguments

    # Early stopping callback
    early_stop = EarlyStopping(monitor='val_loss', patience=2, verbose=1)

    # Custom callback to toggle training status
    toggle_training_status = LambdaCallback(on_epoch_end=lambda epoch, logs: stop_training if stop_training else None)
    
    # Train the model
    model.fit(X_train, y_train_onehot, epochs=10, validation_data=(X_test, y_test_onehot))
    
    # Save the trained model
    model.save(model_path)
    
    # Save the pruned model
    model_for_pruning.save(os.path.join(output_folder, f'Pruned_{Flux_Model}.h5'))
    print("Pruned model training complete and saved!")
    
    # Convert to TFLite
    if not stop_training:
        convert_to_tflite(model)
    
    print("Model training complete and saved!")


def start_training_thread():
    try:
        num_epochs = int(epochs_entry.get())
        Flux_Model = "Flux_Detector_Model"  # You can rename this as you like
        if num_epochs < 1:
            print("Please enter a valid number of epochs.")
            return
        print(f"Training started with {num_epochs} epochs.")
        # Starting the training in a new thread
        training_thread = threading.Thread(target=train_model, args=(num_epochs, Flux_Model))
        training_thread.start()
    except ValueError:
        print("Please enter a valid number for epochs.")


def stop_training():
    global stop_training
    stop_training = True

# Tkinter GUI
window = tk.Tk()
window.title("Flux Stain Detector")

folders = [("With Flux", 'with_flux_display'), ("Without Flux", 'without_flux_display')]
for folder_type, label_name in folders:
    tk.Label(window, text=f"{folder_type} Folder:").pack()
    label_name = tk.Label(window, text="Not Selected")
    label_name.pack()
    tk.Button(window, text="Browse", command=lambda l=label_name, f=folder_type: browse_folder(l, f)).pack()

output_display = tk.Label(window, text="Not Selected")
output_display.pack()
tk.Label(window, text="Output Folder:").pack()
tk.Button(window, text="Browse", command=lambda: browse_output(output_display)).pack()

epochs_entry = tk.Entry(window)
epochs_entry.pack()
tk.Label(window, text="Number of Epochs:").pack()

tk.Button(window, text="Start Training", command=start_training_thread).pack()
tk.Button(window, text="Stop Training", command=stop_training).pack()

window.mainloop()
