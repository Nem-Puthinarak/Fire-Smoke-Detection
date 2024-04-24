import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image, ImageTk
import threading

# Path to your trained model
model_path = r'C:\Users\User\Documents\model-csb\fire_smoke_detection_model.h5'
model = load_model(model_path)

def load_video():
    video_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.avi;*.mov")])
    if video_path:
        start_detection(video_path)

def start_detection(video_path):
    # Clear the canvas
    canvas.delete("all")
    
    # Start the detection in a new thread to keep the GUI responsive
    threading.Thread(target=detect_fire_smoke_video, args=(video_path, model), daemon=True).start()

def detect_fire_smoke_video(video_path, model):
    video = cv2.VideoCapture(video_path)
    ret, frame = video.read()
    if not ret:
        return # No frames to process

    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break # End of video

        # Use the model to detect fire or smoke
        detected_coords = detect_fire_smoke(frame, model)

        if detected_coords:
            # Draw a rectangle around the detected area using the actual coordinates
            x, y, width, height = detected_coords
            cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 0, 255), 2)

        # Convert the frame to a format that can be displayed in Tkinter
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        frame = ImageTk.PhotoImage(frame)

        # Update the canvas with the new frame
        canvas.create_image(0, 0, image=frame, anchor=tk.NW)
        canvas.image = frame # Keep a reference to the image to prevent garbage collection

        # Schedule the next update
        root.after(1, update_canvas, frame)

    video.release()

def update_canvas(frame):
    canvas.create_image(0, 0, image=frame, anchor=tk.NW)
    canvas.image = frame # Keep a reference to the image to prevent garbage collection

def preprocess_frame(frame):
    # Resize the frame to the input size expected by the model
    frame_resized = cv2.resize(frame, (224, 224)) # Adjust based on your model's input size
    # Normalize the frame if necessary (e.g., scale pixel values to [0, 1])
    frame_normalized = frame_resized / 255.0
    return frame_normalized

import numpy as np

def detect_fire_smoke(frame, model):
    # Preprocess the frame
    frame_preprocessed = preprocess_frame(frame)
    # Expand dimensions to match the model's expected input shape (e.g., (batch_size, height, width, channels))
    frame_expanded = np.expand_dims(frame_preprocessed, axis=0)
    
    # Run the model
    predictions = model.predict(frame_expanded)
    
    # Ensure predictions are not empty
    if predictions.shape[0] == 0:
        return None
    
    # Interpret the model's output
    bounding_box = predictions[0] # Assuming the first prediction
    
    # Print predictions and confidence score for debugging
    print("Predictions:", bounding_box)
    confidence_score = bounding_box[-1]
    print("Confidence Score:", confidence_score)
    
    # Adjust threshold based on confidence score distribution in predictions
    if confidence_score > 0.0000000000001: # Lower the threshold
        # Extracting bounding box coordinates
        x_min, y_min, x_max, y_max = bounding_box[:4] # Assuming format [x_min, y_min, x_max, y_max]
        # Converting to integer values for pixel coordinates
        x, y, width, height = int(x_min * frame.shape[1]), int(y_min * frame.shape[0]), \
                              int((x_max - x_min) * frame.shape[1]), int((y_max - y_min) * frame.shape[0])
        print("Bounding Box:", (x, y, width, height))
        return (x, y, width, height)
    else:
        print("Confidence score below threshold")
        return None


# Create the main window
root = tk.Tk()
root.title("Fire and Smoke Detection")

# Create a button to load a video
load_button = tk.Button(root, text="Load Video", command=load_video)
load_button.pack()

# Create a canvas to display the video
canvas = tk.Canvas(root, width=640, height=480) # Initial size, will be adjusted
canvas.pack()

# Start the GUI event loop
root.mainloop()
