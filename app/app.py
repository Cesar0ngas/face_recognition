import streamlit as st
import pandas as pd
import requests
import cv2
import numpy as np
import time
import unicodedata
import re
from pymongo import MongoClient
from datetime import datetime
from PIL import Image
import os

# MongoDB configuration
client = MongoClient("mongodb+srv://cesarcorrea:k9DexhefNDS9GTLs@cluster0.rwqzs.mongodb.net/AttendanceDB?retryWrites=true&w=majority&appName=Cluster0")
db = client.AttendanceDB
students_collection = db.students
attendance_collection = db.attendance

API_URL = "http://localhost:5000/predict"  # URL for the recognition API
RETRAIN_URL = "http://localhost:5000/retrain"  # URL to trigger retraining

DATASET_PATH = "./dataset/train/"  # Path to save student images for training

def predict_image(image):
    try:
        _, img_encoded = cv2.imencode('.jpg', image)
        img_bytes = img_encoded.tobytes()
        files = {"image": ("image.jpg", img_bytes, "image/jpeg")}
        response = requests.post(API_URL, files=files)
        response.raise_for_status()
        result = response.json()
        if "name" in result and "probability" in result:
            return result["name"], result["probability"]
        else:
            return None, None
    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to the API: {e}")
        return None, None

def normalize_string(s):
    s = unicodedata.normalize('NFD', s)
    s = s.encode('ascii', 'ignore').decode("utf-8")
    return s.lower()

def load_students_data():
    students_data = list(students_collection.find({}, {"_id": 0, "name": 1, "matricula": 1, "attendance": 1}))
    return pd.DataFrame(students_data) if students_data else pd.DataFrame(columns=["name", "matricula", "attendance"])

def load_attendance_report():
    today = datetime.now().date()
    today_datetime = datetime.combine(today, datetime.min.time())
    attendance_data = list(attendance_collection.find({"date": today_datetime}, {"_id": 0, "name": 1, "time": 1}))
    return pd.DataFrame(attendance_data) if attendance_data else pd.DataFrame(columns=["name", "time"])

def clear_attendance():
    students_collection.update_many({}, {"$set": {"attendance": False}})
    today = datetime.now()
    attendance_collection.delete_many({"date": {"$gte": today.replace(hour=0, minute=0, second=0, microsecond=0), 
                                                 "$lt": today.replace(hour=23, minute=59, second=59, microsecond=999999)}})
    st.success("Attendance has been cleared successfully.")

def save_image_to_dataset(name, image):
    """Save uploaded image to the dataset directory for the new student."""
    student_path = os.path.join(DATASET_PATH, name)
    os.makedirs(student_path, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    image_path = os.path.join(student_path, f"{timestamp}.jpg")
    image.save(image_path)
    return image_path

def add_student(name, matricula, image):
    """Add a new student and retrain the model with their image."""
    students_collection.insert_one({"name": name, "matricula": matricula, "attendance": False})
    st.success(f"Student {name} added successfully.")
    
    # Save the image to the dataset
    image_path = save_image_to_dataset(name, image)
    
    # Trigger retraining by calling the retrain API
    response = requests.post(RETRAIN_URL)
    
    if response.status_code == 200:
        st.success("Model retrained successfully with the new student.")
    else:
        st.error("Model retraining failed.")

st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose a page", ["Home", "Attendance", "Attendance Report"])

if page == "Home":
    st.image("C:/Users/cesco/Desktop/Personal/UPY/9/didier/2/proyecto/UPY Attendance Systema.jpg", use_column_width=True)
    st.write("Welcome to the UPY Attendance System. Use the sidebar to interact with the application.")

elif page == "Attendance":
    st.title("Attendance System")

    st.sidebar.subheader("Class Information")
    career = st.sidebar.selectbox("Select Career", ["Data Engineer", "Cybersecurity", "Embedded Systems", "Robotics"])
    quarter = st.sidebar.selectbox("Select Quarter", ["Immersion", "Third Quarter", "Sixth Quarter", "Ninth Quarter"])
    group = st.sidebar.selectbox("Select Group", ["A", "B"] if career == "Data Engineer" and quarter == "Ninth Quarter" else [])

    if group == "B":
        df_students = load_students_data()
        if not df_students.empty:
            st.session_state.df_students = df_students
        else:
            st.session_state.df_students = pd.DataFrame(columns=["name", "matricula", "attendance"])

        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("Data loaded successfully for group B:")
            student_table = st.empty()
            student_table.dataframe(st.session_state.df_students.sort_values(by='matricula'))

        with col2:
            st.subheader("Add a new student")
            with st.form("add_student_form"):
                name = st.text_input("Student Name")
                matricula = st.text_input("Student Matricula")
                image_file = st.file_uploader("Upload Student Image", type=["jpg", "jpeg", "png"])
                submitted = st.form_submit_button("Add Student")
                if submitted:
                    if name and matricula and image_file:
                        image = Image.open(image_file)
                        add_student(name, matricula, image)
                        st.experimental_rerun()  # Refresh the app to clear input fields
                    else:
                        st.warning("Please enter the student name, matricula, and upload an image.")

            if st.button("Refresh Table"):
                st.session_state.df_students = load_students_data()
                student_table.dataframe(st.session_state.df_students.sort_values(by='matricula'))

            if st.button("Clear Attendance"):
                clear_attendance()
                st.session_state.df_students = load_students_data()
                student_table.dataframe(st.session_state.df_students.sort_values(by='matricula'))

        st.markdown("---")

        df_attendance = load_attendance_report()
        if not df_attendance.empty:
            st.subheader("Attendance recorded for today:")
            st.dataframe(df_attendance)
        else:
            st.write("No attendance records for today.")

        if st.sidebar.button("Start Facial Recognition"):
            recognition_active = True
            cap = cv2.VideoCapture(0)
            stframe = st.empty()
            status_placeholder = st.sidebar.empty()
            recognized_faces = set()
            last_api_call = time.time()

            status_placeholder.write("Facial recognition activated, waiting for faces...")

            stop_recognition = st.sidebar.button("Stop recognition", key="stop_recognition")

            while recognition_active:
                ret, frame = cap.read()
                if not ret:
                    st.error("Error capturing image from the camera.")
                    break

                stframe.image(frame, channels="BGR", caption="Real-time capture", use_column_width=True)

                if time.time() - last_api_call > 0.5:
                    name, probability = predict_image(frame)
                    last_api_call = time.time()

                    if name and name not in recognized_faces:
                        st.sidebar.success(f"Recognized person: {name} ({probability:.2f}%)")
                        recognized_faces.add(name)
                        status_placeholder.write("Waiting for recognition of new faces...")

                        normalized_name = normalize_string(name)
                        mask = st.session_state.df_students["name"].apply(lambda x: re.search(normalized_name, normalize_string(x)) is not None)
                        st.session_state.df_students.loc[mask, "attendance"] = True

                        for student_name in st.session_state.df_students.loc[mask, "name"]:
                            attendance_collection.insert_one({
                                "name": student_name,
                                "date": datetime.now(),
                                "time": datetime.now()
                            })
                            students_collection.update_one({"name": student_name}, {"$set": {"attendance": True}})
                        
                        student_table.dataframe(st.session_state.df_students.sort_values(by='matricula'))

                if stop_recognition:
                    recognition_active = False
                    cap.release()
                    st.sidebar.write("Facial recognition stopped.")

elif page == "Attendance Report":
    st.title("Attendance Report")

    df_attendance_report = load_attendance_report()
    if not df_attendance_report.empty:
        st.dataframe(df_attendance_report)
    else:
        st.write("No attendance records for today.")
