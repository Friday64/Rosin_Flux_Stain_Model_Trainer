#Create a TensorFlow model that detects flux stains on a PCB
#use threads for all the image processing and training
#using two diffrent folders of images on the desktop
#use image processing for the semi-parent nature of flux, it has a yellow hue with a slite brown tint
#use Tkinter for the GUI
#use root.mainloop() to run the GUI
#ask the user for the 2 input folders "With_flux" and "Without_flux"
#have a button to train the model
#train the model when the button is pressed
#ask user for the output folder to save the model before training the model
#check to see if the model has already exists and reuse it if it does
#create a new model if it does not exist called "Flux_stain_Model.h5"
#save the model to the output folder

#import the necessary libraries first
from tkinter import filedialog
from altair import LabelOverlap
from pydantic import create_model
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
import cv2
import tkinter as tk
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import os
import shutil
import random
import glob
import time

#define the path of the 2 folders that contain the images "With_flux" and "Without_flux"
# Define the path of the 2 folders that contain the images
with_flux_folder = "path/to/With_flux"
without_flux_folder = "path/to/Without_flux"

def preprocess_image(image):
    # Perform preprocessing steps on the image
    # e.g. resize, normalize, etc.
    resized_image = cv2.resize(image, (100, 100))
    normalized_image = resized_image / 255.0
    return normalized_image
    return preprocessed_image

def apply_filter(image, filter_type):
    # Apply a specific filter to the image
    # e.g. blur, sharpen, edge detection, etc.
    if filter_type == 'blur':
        filtered_image = cv2.blur(image, (5, 5))
    elif filter_type == 'sharpen':
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        filtered_image = cv2.filter2D(image, -1, kernel)
    elif filter_type == 'edge_detection':
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray_image, 100, 200)
        filtered_image = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    else:
        raise ValueError("Invalid filter type")
    
    return filtered_image

def enhance_contrast(image):
    # Enhance the contrast of the image
    # e.g. histogram equalization, adaptive histogram equalization, etc.
    contrast_enhanced_image = cv2.equalizeHist(image)

    return contrast_enhanced_image

# Example usage:
image = cv2.imread('path/to/image.jpg')
preprocessed_image = preprocess_image(image)
filtered_image = apply_filter(preprocessed_image, 'blur')
enhanced_image = enhance_contrast(filtered_image)



#create a list called preprocessed_images for the preprocessed images and a list called labels for the labels
preprocessed_images = []
labels = []









def train_model():
    # Ask user for the output folder to save the model
    output_folder = filedialog.askdirectory()

    # Train your model here
    model = create_model()
    

    # Assuming you have preprocessed images stored in a list called preprocessed_images
    # Assuming you have corresponding labels stored in a list called labels

    # Convert the preprocessed_images list to a numpy array
    X = np.array(preprocessed_images)
    Y=np.array(labels)
    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Convert the labels list to a numpy array and encode them as one-hot vectors
    num_classes = len(set(labels))
    y = np.zeros((len(LabelOverlap), num_classes))
    for i, label in enumerate(labels):
        y[i, label] = 1
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))
    
#create main window and widgets
window = tk.Tk()
window.title("Flux Stain Classifier")
window.geometry("500x500")
#ask the user for the 2 input folders "With_flux" and "Without_flux" in the gui
import tkinter as tk
from tkinter import filedialog


#create main window and widgets
window = tk.Tk()
window.title("Flux Stain Classifier")
window.geometry("500x500")

# Create entry fields for the input folders
with_flux_entry = tk.Entry(window)
with_flux_entry.pack()

without_flux_entry = tk.Entry(window)
without_flux_entry.pack()

# Create a button to train the model
train_button = tk.Button(window, text="Train Model", command=train_model)
train_button.pack()

window.mainloop()























































