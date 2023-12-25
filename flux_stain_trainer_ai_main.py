import tkinter as tk
from tkinter import messagebox
import subprocess

# Tkinter UI setup
def start_training():
    subprocess.Popen(["python", "train.py"])
    messagebox.showinfo("Training Started", "Training process has started.")

window = tk.Tk()
window.title("Flux Stain Detector")

tk.Label(window, text="Number of Epochs:").pack()
epochs_entry = tk.Entry(window)
epochs_entry.pack()
train_button = tk.Button(window, text="Train Model", command=start_training)
train_button.pack()

window.mainloop()
