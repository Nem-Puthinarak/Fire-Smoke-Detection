import streamlit as st
import threading
import sys
import cv2  # Import OpenCV library for video processing
from ultralytics import YOLO
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from keys import sender_pass, senderEmail

# Global flag for stopping the detection process
stop_detection = False

# Streamlit app
st.title("Fire-Smoke_detection")

# Button to exit the script
if st.button('Exit'):
    st.warning("Exiting the script...")
    stop_detection = True  # Signal the detection process to stop
    sys.exit()  # Terminate the program

# Radio button for detection method
detection_method = st.radio(
    "Select the detection method:",
    ("CCTV", "Upload Video")
)

# Function to send email alert
def send_alert_email(receiver_email):
    st.write("Sending alert email...")  # Debug print
    # Set up your email details
    sender_email =  senderEmail
    sender_password = sender_pass  

    # Create the email
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = "Fire or Smoke Detected"
    body = "Fire or smoke has been detected in the video."
    msg.attach(MIMEText(body, 'plain'))

    # Send the email
    try:
        st.write("Attempting to send email...")  # Debug print
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, sender_password)
        text = msg.as_string()
        server.sendmail(sender_email, receiver_email, text)
        server.quit()
        st.success("Alert email sent successfully!")
    except Exception as e:
        st.error(f"Error sending email: {e}")

# Function to run detection on uploaded video
def run_detection(video_path, detection_option, receiver_email):
    global stop_detection
    # Choose the model based on the user's selection
    if detection_option == "Fire":
        model = YOLO("fire-smoke.pt")
    elif detection_option == "Smoke":
        model = YOLO("smoke.pt")  
    else:
        model = YOLO("fire-smoke.pt")  # Default to both if "Both" is selected

    # Use the YOLO model to predict on the uploaded video
    results = model.predict(source=video_path, imgsz=640, save=True, show=True)

    fire_detected = False
    smoke_detected = False
    for result in results:
        if 'fire' in result['labels']:
            fire_detected = True
        if 'smoke' in result['labels']:
            smoke_detected = True

    # Debugging: Print detection results
    st.write(f"Fire detected: {fire_detected}")
    st.write(f"Smoke detected: {smoke_detected}")

    # If fire or smoke is detected, send the alert email
    if (fire_detected and detection_option in ["Fire", "Both"]) or (
            smoke_detected and detection_option in ["Smoke", "Both"]):
        send_alert_email(receiver_email)
    else:
        st.info("No fire or smoke detected in the video.")

# Handle CCTV option
if detection_method == "CCTV":
    # Input field for the RTSP URL
    cctv_ip = st.text_input("Enter the RTSP URL of the CCTV camera:", "")

    # Input field for the receiver's email address
    receiver_email = st.text_input("Enter the receiver's email address:", "")

    # Select box for model selection
    model_selection = st.selectbox(
        "Choose the model for detection:",
        ("Fire", "Smoke", "Both")
    )

    # Extract username, password, and IP address from the RTSP URL
    if "rtsp://" in cctv_ip:
        username_password, rest_of_url = cctv_ip.split('@')[0].split('//')[1], cctv_ip.split('@')[1]
        username, password = username_password.split(':')[0], username_password.split(':')[1]
        cctv_ip = rest_of_url.split(':')[0]

    if cctv_ip and receiver_email:
        # Start the detection process in a separate thread
        detection_thread = threading.Thread(target=run_detection, args=(cctv_ip, model_selection, receiver_email))
        detection_thread.start()

        # Wait for the detection thread to finish
        detection_thread.join()
    else:
        st.warning("Please enter the RTSP URL of the CCTV camera and the receiver's email address.")

# Handle Upload Video option
if detection_method == "Upload Video":
    # Upload video file
    uploaded_file = st.file_uploader("Upload your video file", type=["mp4"])

    # Input field for the receiver's email address
    receiver_email = st.text_input("Enter the receiver's email address:", "")

    # Select box for detection preference
    detection_option = st.selectbox(
        "Choose the object to detect:",
        ("Fire", "Smoke", "Both")
    )

    if uploaded_file is not None and receiver_email:
        # Save the uploaded file temporarily
        video_path = "uploaded_video.mp4"
        with open(video_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        # Start the detection process in a separate thread
        detection_thread = threading.Thread(target=run_detection, args=(video_path, detection_option, receiver_email))
        detection_thread.start()

        # Wait for the detection thread to finish
        detection_thread.join()
    else:
        st.warning("Please upload a video file and enter the receiver's email address.")
