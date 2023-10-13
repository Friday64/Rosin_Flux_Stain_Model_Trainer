import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import threading
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Globals for folder paths
with_flux_folder = None
without_flux_folder = None
output_folder = None

# Variable to stop training
stop_training = False

# Function to stop training
def stop():
    global stop_training
    stop_training = True

# Function to train the model
def train_model():
    global stop_training
    stop_training = False
    
    # Data augmentation with RGB values
    train_datagen = ImageDataGenerator(
        rescale=1.0/255,
        brightness_range=[0.2, 1.0],
        rotation_range=40,
        shear_range=0.2,
        zoom_range=0.2,
        fill_mode='nearest',
        horizontal_flip=True
    )
    
    # Prepare training and validation data
    train_generator = train_datagen.flow_from_directory(
        with_flux_folder,
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary'
    )
    
    validation_generator = train_datagen.flow_from_directory(
        without_flux_folder,
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary'
    )
    
    # Build the model
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    history = model.fit(
        train_generator,
        epochs=10,
        verbose=1,
        validation_data=validation_generator,
        callbacks=[lambda epoch, logs: stop_training]  # Early stopping
    )
    
# Function to select with_flux folder
def select_with_flux_folder():
    global with_flux_folder
    folder_selected = filedialog.askdirectory()
    with_flux_folder = folder_selected

# Function to select without_flux folder
def select_without_flux_folder():
    global without_flux_folder
    folder_selected = filedialog.askdirectory()
    without_flux_folder = folder_selected

# Tkinter GUI
root = tk.Tk()
root.title("Flux Stain Trainer")

frame = ttk.Frame(root, padding="10")
frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

button_with_flux_folder = ttk.Button(frame, text="Select With-Flux Folder", command=select_with_flux_folder)
button_with_flux_folder.grid(row=0, column=0, sticky=tk.W)

button_without_flux_folder = ttk.Button(frame, text="Select Without-Flux Folder", command=select_without_flux_folder)
button_without_flux_folder.grid(row=0, column=1, sticky=tk.W)

button_start = ttk.Button(frame, text="Start Training", command=lambda: threading.Thread(target=train_model).start())
button_start.grid(row=0, column=2, sticky=tk.W)

button_stop = ttk.Button(frame, text="Stop Training", command=stop)
button_stop.grid(row=0, column=3, sticky=tk.W)

root.mainloop()
