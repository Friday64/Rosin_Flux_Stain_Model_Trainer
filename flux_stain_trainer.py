# Standard library imports
import logging
import os

# Related third-party imports
import tensorflow as tf
from tensorflow import keras
from keras.applications import MobileNetV2
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# PyQt5 imports for GUI
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QLineEdit, QPushButton, QMessageBox

# Configure logging
logging.basicConfig(level=logging.INFO)

# Constants for paths and hyperparameters
WITH_FLUX_FOLDER = "C:/Users/Matthew/Desktop/Programming/Detect_Flux_Project/Flux_Data/With_Flux"
WITHOUT_FLUX_FOLDER = "C:/Users/Matthew/Desktop/Programming/Detect_Flux_Project/Flux_Data/Without_Flux"
MODEL_PATH = "C:/Users/Matthew/Desktop/Programming/Detect_Flux_Project/Flux_Models/flux_model_tf"
TFLITE_MODEL_PATH = "C:/Users/Matthew/Desktop/Programming/Detect_Flux_Project/Flux_Models/flux_model_tf_lite"
IMG_SIZE = (256, 256)
LEARNING_RATE = 0.1
BATCH_SIZE = 32

#Function to check and create folders and create them if they don't exist
def check_and_create_folders(MODEL_PATH, TFLITE_MODEL_PATH):
    model_folder = os.path.dirname(MODEL_PATH)
    tflite_folder = os.path.dirname(TFLITE_MODEL_PATH)
    folders_to_check = [model_folder, tflite_folder]

    for folder in folders_to_check:
        if not os.path.exists(folder):
            try:
                os.makedirs(folder)
                logging.info(f"Folder '{folder}' created.")
            except Exception as e:
                logging.error(f"Error creating folder '{folder}': {e}")
        else:
            logging.info(f"Folder '{folder}' already exists.")

# Add this function call at the beginning of your code
check_and_create_folders(MODEL_PATH, TFLITE_MODEL_PATH)

# Enable mixed precision training
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Load data paths and labels
def load_data_paths():
    all_data = []
    all_labels = []
    for folder, label in [(WITH_FLUX_FOLDER, 1), (WITHOUT_FLUX_FOLDER, 0)]:
        for filename in os.listdir(folder):
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                all_data.append(os.path.join(folder, filename))
                all_labels.append(label)
    return all_data, all_labels

# Transfer Learning model creation function
def create_model():
    base_model = MobileNetV2(input_shape=(256, 256, 3), include_top=False, weights='imagenet')
    base_model.trainable = False

    model = keras.Sequential([
        base_model,
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(2, activation='softmax')
    ])

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    model.summary()  # Print the model structure

    return model

# Data augmentation
def get_data_augmentation():
    return ImageDataGenerator(
        rotation_range=20,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest"
    )

# Preprocess and load data using tf.data
def preprocess_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMG_SIZE)
    image = keras.applications.mobilenet_v2.preprocess_input(image)
    return image

def load_dataset(data_paths, labels, batch_size):
    path_ds = tf.data.Dataset.from_tensor_slices(data_paths)
    image_ds = path_ds.map(preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(labels, tf.int64))
    dataset = tf.data.Dataset.zip((image_ds, label_ds))
    return dataset.shuffle(len(data_paths)).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

# Function to train the model
def train_model(model, train_ds, val_ds, epochs):
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs
    )
    return history

# Function to save the model
def save_model(model, model_path):
    model.save(model_path)

# Function to load or create model
def load_or_create_model(model_path):
    if os.path.exists(model_path):
        logging.info("Loading existing model.")
        return keras.models.load_model(model_path)
    else:
        logging.info("Creating new model.")
        return create_model()

# Function to convert the model to TensorFlow Lite
def convert_to_tflite(model_path, tflite_model_path, quantize=False):
    model = keras.models.load_model(model_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    with open(tflite_model_path, 'wb') as f:
        f.write(tflite_model)
    logging.info(f'Model converted to TensorFlow Lite and saved to {tflite_model_path}')

# PyQt5 GUI setup for training
class TrainingWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
    
    def initUI(self):
        self.setWindowTitle('Flux Stain Detector Training')
        layout = QVBoxLayout()

        label = QLabel('Number of Epochs:')
        self.epochsEntry = QLineEdit()
        self.trainButton = QPushButton('Train Model')
        self.trainButton.clicked.connect(self.startTraining)

        layout.addWidget(label)
        layout.addWidget(self.epochsEntry)
        layout.addWidget(self.trainButton)

        self.setLayout(layout)
        self.show()

    def startTraining(self):
        self.trainButton.setEnabled(False)
        epochs = self.epochsEntry.text()
        if not epochs.isdigit():
            QMessageBox.critical(self, "Error", "Please enter a valid number of epochs.")
            self.trainButton.setEnabled(True)
            return

        try:
            model = load_or_create_model(MODEL_PATH)
            train_ds, val_ds = self.prepareData()
            history = train_model(model, train_ds, val_ds, int(epochs))
            save_model(model, MODEL_PATH)

            # Convert and save the TensorFlow Lite model
            convert_to_tflite(MODEL_PATH, TFLITE_MODEL_PATH, quantize=True)

            QMessageBox.information(self, "Training Complete", "Model trained and saved successfully.")
        except Exception as e:
            QMessageBox.critical(self, "Training Error", f"An error occurred: {str(e)}")
            logging.error("Training Error", exc_info=True)

        self.trainButton.setEnabled(True)

    def prepareData(self):
        all_data, all_labels = load_data_paths()
        train_data, val_data, train_labels, val_labels = train_test_split(all_data, all_labels, test_size=0.2)
        train_ds = load_dataset(train_data, train_labels, BATCH_SIZE)
        val_ds = load_dataset(val_data, val_labels, BATCH_SIZE)
        return train_ds, val_ds

def main():
    app = QApplication([])
    ex = TrainingWindow()
    app.exec_()

if __name__ == "__main__":
    main()
