# Import required libraries
from PIL import Image
import gradio as gr
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Define the data directory path (you can modify this)
data_dir = "C:/Users/Matthew/Desktop/Flux_Stain_Project_Pics/Flux_Pics"

# Define ImageDataGenerator
datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# Define model architecture
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

# Initialize empty lists to hold images and labels
collected_images = []
collected_labels = []

def preprocess_image(image):
    # Resize the image to (150, 150)
    image = Image.fromarray(image.astype('uint8'), 'RGB')
    image = image.resize((150, 150))
    image_array = np.array(image)
    
    # Normalize to [0,1]
    image_array = image_array / 255.0
    
    # Expand dimensions for batch size
    image_array = np.expand_dims(image_array, axis=0)
    
    return image_array
def train_on_batch(images, labels):
    # Reshape the data if necessary
    images = np.vstack(images)  # Stack arrays vertically
    labels = np.array(labels)   # Convert labels list to array
    
    # Train the model on this batch
    history = model.train_on_batch(images, labels)
    
    # Print or update metrics (you can expand this)
    print(f"Training metrics: {history}")

# Gradio UI function
def classify_image(image, choice):
    global collected_images, collected_labels

    # Preprocess image and label
    preprocessed_image = preprocess_image(image)
    label = 1 if choice == 'With Flux' else 0
    
    # Collect for batch
    collected_images.append(preprocessed_image)
    collected_labels.append(label)

    # Train in batches
    if len(collected_images) >= 32:
        train_on_batch(np.array(collected_images), np.array(collected_labels))
        collected_images, collected_labels = [], []

    return f"Image processed and labeled as {'With Flux' if label == 1 else 'Without Flux'}"


# Define the Gradio interface
iface = gr.Interface(
    fn=classify_image, 
    inputs=["image", gr.inputs.Radio(["With Flux", "Without Flux"])], 
    outputs="text"
)

# Launch the Gradio interface
iface.launch()
