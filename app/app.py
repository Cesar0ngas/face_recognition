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

API_URL = "https://peaceful-castle-86006.herokuapp.com/predict"
DATASET_PATH = "./dataset/train/" 

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

def add_student(name, matricula):
    students_collection.insert_one({"name": name, "matricula": matricula, "attendance": False})
    st.success(f"Student {name} added successfully.")

st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose a page", ["Home", "Attendance", "Attendance Report"])

if page == "Home":
    st.write("Welcome to the UPY Attendance System. Use the sidebar to interact with the application.")

elif page == "Attendance":
    st.title("Attendance System")

    st.sidebar.subheader("Class Information")
    career = st.sidebar.selectbox("Select Career", ["Data Engineer", "Cybersecurity", "Embedded Systems", "Robotics"])
    quarter = st.sidebar.selectbox("Select Quarter", ["Immersion", "Third Quarter", "Sixth Quarter", "Ninth Quarter"])
    group = st.sidebar.selectbox("Select Group", ["A", "B"] if career == "Data Engineer" and quarter == "Ninth Quarter" else [])

    if group == "B":
        df_students = load_students_data()
        st.session_state.df_students = df_students if not df_students.empty else pd.DataFrame(columns=["name", "matricula", "attendance"])

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
                submitted = st.form_submit_button("Add Student")
                if submitted and name and matricula:
                    add_student(name, matricula)
                    st.session_state.df_students = load_students_data()
                elif submitted:
                    st.warning("Please enter the student name and matricula.")

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

        st.subheader("Facial Recognition")
        
        # Capture image with camera_input
        image = st.camera_input("Capture your image")

        if image:
            img = Image.open(image)
            img_array = np.array(img)
            name, probability = predict_image(img_array)

            if name:
                st.success(f"Recognized person: {name} ({probability:.2f}%)")
                st.session_state.df_students.loc[st.session_state.df_students["name"] == name, "attendance"] = True
                attendance_collection.insert_one({"name": name, "date": datetime.now(), "time": datetime.now()})
                students_collection.update_one({"name": name}, {"$set": {"attendance": True}})
                student_table.dataframe(st.session_state.df_students.sort_values(by='matricula'))

