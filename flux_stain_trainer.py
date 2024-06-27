import logging
import os
import tensorflow as tf
from tensorflow import keras
from keras.applications import MobileNetV2
from keras.preprocessing.image import ImageDataGenerator
from dotenv import load_dotenv
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QLineEdit, QPushButton, QMessageBox
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load environment variables
load_dotenv()

# Constants for paths and hyperparameters
WITH_FLUX_FOLDER = os.getenv("WITH_FLUX_FOLDER")
WITHOUT_FLUX_FOLDER = os.getenv("WITHOUT_FLUX_FOLDER")
MODEL_PATH = os.getenv("MODEL_PATH")
IMG_SIZE = (256, 256)
LEARNING_RATE = 0.001  # Adjusted learning rate
BATCH_SIZE = 32

# Function to check and create folders
def check_and_create_folders(*folders: str) -> None:
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
        logging.info(f"Checked/created folder: {folder}")

check_and_create_folders(WITH_FLUX_FOLDER, WITHOUT_FLUX_FOLDER, os.path.dirname(MODEL_PATH))

# Load data paths and labels
def load_data_paths() -> 'tuple[list[str], list[int]]':
    all_data, all_labels = [], []
    for folder, label in [(WITH_FLUX_FOLDER, 1), (WITHOUT_FLUX_FOLDER, 0)]:
        for filename in os.listdir(folder):
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                all_data.append(os.path.join(folder, filename))
                all_labels.append(label)
    return all_data, all_labels

# Transfer Learning model creation function
def create_model() -> keras.Model:
    base_model = MobileNetV2(input_shape=(256, 256, 3), include_top=False, weights='imagenet')
    base_model.trainable = True
    for layer in base_model.layers[:-10]:
        layer.trainable = False

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
    return model

# Data augmentation
def get_data_augmentation() -> ImageDataGenerator:
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
def preprocess_image(image_path: str) -> tf.Tensor:
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMG_SIZE)
    image = keras.applications.mobilenet_v2.preprocess_input(image)
    return image

def load_dataset(data_paths: 'list[str]', labels: 'list[int]', batch_size: int) -> tf.data.Dataset:
    path_ds = tf.data.Dataset.from_tensor_slices(data_paths)
    image_ds = path_ds.map(preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(labels, tf.int64))
    dataset = tf.data.Dataset.zip((image_ds, label_ds))
    return dataset.shuffle(len(data_paths)).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

# Function to load or create model
def load_or_create_model() -> keras.Model:
    if os.path.exists(MODEL_PATH):
        try:
            logging.info("Loading existing model.")
            return keras.models.load_model(MODEL_PATH)
        except OSError as e:
            logging.error(f"Error loading model: {e}. Creating a new model.")
    logging.info("Creating new model.")
    model = create_model()
    model.save(MODEL_PATH)
    return model

# PyQt5 GUI setup for training
class TrainingWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
    
    def initUI(self) -> None:
        self.setWindowTitle('Flux Stain Detector Training')
        layout = QVBoxLayout()
        layout.addWidget(QLabel('Number of Epochs:'))
        self.epochsEntry = QLineEdit()
        layout.addWidget(self.epochsEntry)
        self.trainButton = QPushButton('Train Model')
        self.trainButton.clicked.connect(self.startTraining)
        layout.addWidget(self.trainButton)
        self.setLayout(layout)
        self.show()

    def startTraining(self) -> None:
        self.trainButton.setEnabled(False)
        epochs = self.epochsEntry.text()
        if not epochs.isdigit():
            QMessageBox.critical(self, "Error", "Please enter a valid number of epochs.")
            self.trainButton.setEnabled(True)
            return

        try:
            model = load_or_create_model()
            all_data, all_labels = load_data_paths()
            train_ds, val_ds = self.prepareData(all_data, all_labels)
            model.fit(train_ds, validation_data=val_ds, epochs=int(epochs))
            model.save(MODEL_PATH)
            QMessageBox.information(self, "Training Complete", "Model trained and saved successfully.")
        except Exception as e:
            QMessageBox.critical(self, "Training Error", f"An error occurred: {str(e)}")
            logging.error("Training Error", exc_info=True)
        self.trainButton.setEnabled(True)

    def prepareData(self, all_data: 'list[str]', all_labels: 'list[int]') -> 'tuple[tf.data.Dataset, tf.data.Dataset]':
        train_data, val_data, train_labels, val_labels = train_test_split(all_data, all_labels, test_size=0.2)
        train_ds = load_dataset(train_data, train_labels, BATCH_SIZE)
        val_ds = load_dataset(val_data, val_labels, BATCH_SIZE)
        return train_ds, val_ds

def main() -> None:
    app = QApplication([])
    ex = TrainingWindow()
    app.exec_()

if __name__ == "__main__":
    main()
