import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
from keras.models import load_model
import os

# Load your pre-trained model
# Replace 'your_model.h5' with your model file
model = load_model(
    'Model___Date_Time_2023_10_14__12_05_02___Loss_0.0162440687417984___Accuracy_0.9964062571525574.h5')

# Class labels
class_labels = ["WalkingWithDog", "TaiChi", "Swing",
                "HorseRace"]  # Update with your class labels

# Function to preprocess and classify a frame


def classify_frame(frame):
    frame = cv2.resize(frame, (64, 64))
    frame = frame / 255.0
    frame = np.expand_dims(frame, axis=0)
    predictions = model.predict(frame)
    predicted_class = class_labels[np.argmax(predictions)]
    return predicted_class

# Function to open a video file and display predictions


def open_video_file():
    file_path = filedialog.askopenfilename(
        filetypes=[("Video files", "*.avi")])
    if file_path:
        cap = cv2.VideoCapture(file_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Classify the frame
            predicted_class = classify_frame(frame)

            # Display the frame and predicted activity
            cv2.putText(frame, predicted_class, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow('Video', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


# Create the main GUI window
root = tk.Tk()
root.title("Video Activity Classifier")

# Create a button to open a video file
open_button = tk.Button(root, text="Open Video File", command=open_video_file)
open_button.pack()

root.mainloop()
