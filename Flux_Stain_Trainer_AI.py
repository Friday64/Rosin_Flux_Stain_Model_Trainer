# Standard library imports
import os  # For operating system dependent functionality

# Third-party libraries for numerical operations and array handling
import numpy as np  # NumPy for numerical operations
import cv2  # OpenCV for computer vision tasks

# Deep learning and neural network frameworks
import tensorflow as tf  # TensorFlow for machine learning and neural network operations
from keras.models import Sequential  # Sequential for a linear stack of neural network layers
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D  # Various layers for neural networks
from keras.callbacks import EarlyStopping  # EarlyStopping to stop training when a monitored metric stops improving
from keras.utils import to_categorical  # to_categorical for converting labels to one-hot encoded format
from keras.optimizers import Adam  # Adam optimizer for training neural networks

# Machine learning utilities
from sklearn.model_selection import train_test_split  # train_test_split to split datasets into training and test sets

# GUI application library
import tkinter as tk  # tkinter for GUI applications

# Paths to image folders
with_flux_folder = "C:/Users/Matthew/Desktop/Programming/Detect_Flux_Project/Flux_Data/With_Flux"
without_flux_folder = "C:/Users/Matthew/Desktop/Programming/Detect_Flux_Project/Flux_Data/Without_Flux"
output_folder = "C:/Users/Matthew/Desktop/Programming/Detect_Flux_Project/Flux_Models"

# Size to which images will be resized
img_size = (128, 128)

# Global flag to control training
stop_training = False

# Custom callback to stop training
class CustomStopTrainingCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        global stop_training
        if stop_training:
            print("Stopping training...")
            self.model.stop_training = True

# Function to preprocess and load images
def preprocess_and_load_images(directory_path, img_size):
    """
    Preprocesses and loads images from a given directory path.
    Images are converted to grayscale and resized to img_size.

    :param directory_path: Path to the directory containing images.
    :param img_size: Tuple representing the size to resize images to.
    :return: A numpy array of preprocessed images.
    """

    # Get a list of image file paths from the directory
    image_files = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # Preallocate a numpy array for the dataset for efficiency
    dataset = np.zeros((len(image_files), img_size[0], img_size[1], 1), dtype=np.float32)

    # Loop over the image files using enumerate to keep an index
    for idx, image_file in enumerate(image_files):
        try:
            # Read the image as grayscale
            img = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)

            # Check if the image was loaded properly
            if img is not None:
                # Resize the image to the specified size
                img = cv2.resize(img, img_size)

                # Normalize the image data to 0-1 range
                img = img / 255.0

                # Add the processed image to the dataset
                dataset[idx, :, :, 0] = img  # Use the last dimension for channel

            else:
                # Log an error message if the image couldn't be loaded
                print(f"Warning: Image {image_file} could not be loaded and will be skipped.")

        except Exception as e:
            # Log an error message if something goes wrong
            print(f"Error processing image {image_file}: {e}")

    # Log the completion of the loading process
    print(f"Finished loading and preprocessing {len(dataset)} images from {directory_path}")

    # Return the preprocessed dataset
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

# Function to train the model
def train_model(epochs, model, callbacks_list):
    callbacks_list.append(CustomStopTrainingCallback())
    # Ensure 'model' is an instance of a Keras model
    if not isinstance(model, tf.keras.Model):
        raise ValueError("The 'model' argument must be an instance of a Keras model.")

    # Preprocess the images for training and testing
    train_data = preprocess_and_load_images(with_flux_folder, (128, 128))
    train_labels = np.ones(train_data.shape[0])
    test_data = preprocess_and_load_images(without_flux_folder, (128, 128))
    test_labels = np.zeros(test_data.shape[0])

    # Concatenate train and test data for splitting
    data = np.vstack([train_data, test_data])
    labels = np.hstack([train_labels, test_labels])

    # Split into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

    # Convert labels to categorical (one-hot encoding)
    y_train = to_categorical(y_train, num_classes=2)
    y_test = to_categorical(y_test, num_classes=2)

    try:
        # Train the model
        history = model.fit(
            x_train, y_train,
            validation_data=(x_test, y_test),
            epochs=epochs,
            batch_size=64,
            callbacks=callbacks_list,
            verbose=1
        )

        # Check if training was successful
        if not stop_training:
            # Save the trained model
            model.save(f"{output_folder}/flux_model.h5")
            print("Model trained and saved successfully.")

        else:
            print("Training was stopped prematurely.")
    except Exception as e:
        print(f"An error occurred during training: {e}")

# Function to stop training
def halt_training():
    global stop_training
    stop_training = True  # This will be checked within the training loop or callback to stop the training

def start_training():
    try:
        epochs = int(epochs_entry.get())  # Ensure this is a valid integer
    except ValueError:
        print("Number of epochs is not a valid integer.")
        return

    # Path to the model file
    model_file_path = f"{output_folder}/flux_model.h5"

    # Check if the model already exists
    if os.path.exists(model_file_path):
        print("Loading an existing model for retraining...")
        model = tf.keras.models.load_model(model_file_path)
    else:
        print("Creating a new model...")
        model = create_model(input_shape=(128, 128, 1), num_classes=2)  # Adjust input_shape as per your data

    callbacks_list = [EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)]
    train_model(epochs, model, callbacks_list)

# Initialize Tkinter window
window = tk.Tk()
window.title("Flux Stain Detector")

# Create UI elements
tk.Label(window, text="Number of Epochs:").pack()
epochs_entry = tk.Entry(window)
epochs_entry.pack()

train_button = tk.Button(window, text="Train Model", command=start_training)
train_button.pack()
stop_button = tk.Button(window, text="Stop Training", command=halt_training)
stop_button.pack()

# Run the Tkinter event loop
window.mainloop()
