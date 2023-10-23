import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical


# Global Variables
with_flux_folder = ""
without_flux_folder = ""
output_folder = "output"

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
def preprocess_and_load_images(folder, label):
    """
    Loads images from the folder, preprocesses them, and returns as an array.
    
    Parameters:
        folder (str): The folder path where the images are stored.
        label (int): The label to be used for these images.
        
    Returns:
        list: A list of tuples where the first element is the image array and the second is the label.
    """
    # Validate the folder exists
    if not os.path.exists(folder):
        print(f"Folder {folder} does not exist.")
        return []

    processed_images = []
    
    # Loop through all jpg files in the folder
    for filename in os.listdir(folder):
        if filename.endswith(".jpg"):
            # Read and resize the image
            img = cv2.imread(os.path.join(folder, filename))
            img_resized = cv2.resize(img, (100, 100))
            
            # Normalize pixel values
            img_normalized = img_resized / 255.0
            
            processed_images.append((img_normalized, label))
            
    return processed_images

def create_model():
    """
    Creates a Convolutional Neural Network (CNN) model for classifying images with or without flux stains.

    Returns:
        keras.models.Sequential: The CNN model
    """
    # Initialize the model
    model = Sequential()

    # Add layers
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    
    # Output layer
    model.add(Dense(2, activation='softmax'))

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def train_model(epochs):
    # Prepare the image data
    with_flux_data = preprocess_and_load_images(with_flux_folder, 1)
    without_flux_data = preprocess_and_load_images(without_flux_folder, 0)
    
    # Combine both types of data
    all_data = with_flux_data + without_flux_data
    all_images, all_labels = zip(*all_data)

    # Convert to NumPy arrays
    all_images = np.array(all_images)
    all_labels = np.array(all_labels)
    
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(all_images, all_labels, test_size=0.2)

    # Convert labels to categorical
    y_train, y_test = to_categorical(y_train, 2), to_categorical(y_test, 2)
    
    # Create or load the model
    model_path = 'Flux_Stain_Model.h5'
    model = load_model(model_path) if os.path.exists(model_path) else create_model()

    # Data augmentation
    datagen = ImageDataGenerator(rotation_range=20, width_shift_range=0.2,
                                 height_shift_range=0.2, horizontal_flip=True)
    
    # Fit the data generator
    datagen.fit(X_train)
    
    # Early stopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=50)
    
    # Train the model
    model.fit(datagen.flow(X_train, y_train, batch_size=32),
              epochs=epochs,
              validation_data=(X_test, y_test),
              callbacks=[early_stopping])
    
    # Save the trained model
    model.save(os.path.join(output_folder, model_path))
    print("Model training complete and saved!")

def start_training():
    try:
        # Read the number of epochs from the Entry widget
        num_epochs = int(epochs_entry.get())
        
        # Validate the number of epochs
        if num_epochs < 1:
            print("Please enter a valid number of epochs.")
            return

        # Call the function to train the model
        train_model(num_epochs)
        print("Training started with {} epochs.".format(num_epochs))
    except ValueError:
        print("Please enter a valid number for epochs.")

# Tkinter GUI
window = tk.Tk()
window.title("Flux Stain Detector")

# Folders for images with flux and without flux
folders = [("With Flux", 'with_flux_display'), ("Without Flux", 'without_flux_display')]

for folder_type, label_name in folders:
    tk.Label(window, text=f"{folder_type} Folder:").pack()
    label_name = tk.Label(window, text="Not Selected")
    label_name.pack()
    tk.Button(window, text="Browse", command=lambda l=label_name, f=folder_type: browse_folder(l, f)).pack()

# Output Folder
output_display = tk.Label(window, text="Not Selected")
output_display.pack()
tk.Label(window, text="Output Folder:").pack()
tk.Button(window, text="Browse", command=lambda: browse_output(output_display)).pack()

# Number of Epochs
epochs_entry = tk.Entry(window)
epochs_entry.pack()
tk.Label(window, text="Number of Epochs:").pack()

# Button to Start Training
tk.Button(window, text="Start Training", command=start_training).pack()

window.mainloop()
