from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
import face_recognition
import base64
import sqlite3
from datetime import datetime
import os

app = Flask(__name__)


# Initialize database
def init_db():
    conn = sqlite3.connect('attendance.db')
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS attendance
                      (name TEXT, date TEXT, time TEXT)''')
    conn.commit()
    conn.close()


# Add an attendance record
def add_attendance_record(name):
    conn = sqlite3.connect('attendance.db')
    cursor = conn.cursor()
    date, time = datetime.now().strftime("%Y-%m-%d"), datetime.now().strftime("%H:%M:%S")
    cursor.execute("INSERT INTO attendance (name, date, time) VALUES (?, ?, ?)", (name, date, time))
    conn.commit()
    conn.close()
    write_attendance_to_file(name, date, time)  # Write to file after adding record


# Write attendance record to a file
def write_attendance_to_file(name, date, time):
    with open('attendance_records.txt', 'a') as f:
        f.write(f"{name} - Date: {date}, Time: {time}\n")


# Load and encode student images
known_face_encodings = []
known_face_names = []


def load_student_images():
    for filename in os.listdir('student_images'):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            name = filename.split('.')[0]
            image_path = f'student_images/{filename}'
            image = face_recognition.load_image_file(image_path)
            encoding = face_recognition.face_encodings(image)[0]
            known_face_encodings.append(encoding)
            known_face_names.append(name)


# Initialize database and load student images on startup
init_db()
load_student_images()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/capture', methods=['POST'])
def capture():
    try:
        data = request.json['image']
        image_data = base64.b64decode(data.split(',')[1])
        np_array = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

        # Detect and recognize faces
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)

        attendance_list = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            if True in matches:
                match_index = matches.index(True)
                name = known_face_names[match_index]
                if name not in attendance_list:  # Avoid duplicate entries
                    attendance_list.append(name)
                    add_attendance_record(name)

        response_message = "Attendance recorded for: " + ', '.join(
            attendance_list) if attendance_list else "No recognized students found."
        return jsonify({'message': response_message})
    except Exception as e:
        return jsonify({'message': f"Error processing image: {str(e)}"})


@app.route('/upload', methods=['POST'])
def upload():
    try:
        # Retrieve the image data from the request
        data = request.json['image']
        image_data = base64.b64decode(data.split(',')[1])
        np_array = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

        # Detect and recognize faces
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)

        attendance_list = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            if True in matches:
                match_index = matches.index(True)
                name = known_face_names[match_index]
                if name not in attendance_list:  # Avoid duplicate entries
                    attendance_list.append(name)
                    add_attendance_record(name)

        response_message = "Attendance recorded for: " + ', '.join(
            attendance_list) if attendance_list else "No recognized students found."
        return jsonify({'message': response_message})
    except Exception as e:
        return jsonify({'message': f"Error processing uploaded image: {str(e)}"})


@app.route('/attendance')
def attendance():
    conn = sqlite3.connect('attendance.db')
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM attendance")
    records = cursor.fetchall()
    conn.close()
    return jsonify(records)


if __name__ == '__main__':
    app.run(debug=True)

