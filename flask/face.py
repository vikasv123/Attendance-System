from flask import *
import face_recognition
import cv2
import numpy as np
import csv
from datetime import datetime

app = Flask(__name__)

@app.route('/at')
def attend():
    # Face recognition variables
    known_faces_names = ["Sarwan Sir","Vikas","Anita Maam"]
    known_face_encodings = []

    # Load known face encodings
    sir_image = face_recognition.load_image_file("photos/sir.jpeg")
    sir_encoding = face_recognition.face_encodings(sir_image)[0]

    vikas_image = face_recognition.load_image_file("photos/vikas.jpg")
    vikas_encoding = face_recognition.face_encodings(vikas_image)[0]

    maam_image = face_recognition.load_image_file("photos/maam.png")
    maam_encoding = face_recognition.face_encodings(maam_image)[0]


    known_face_encodings = [sir_encoding,vikas_encoding,maam_encoding]

    students = known_faces_names.copy()

    face_locations = []
    face_encodings = []
    face_names = []

    now = datetime.now()
    current_date = now.strftime("%Y-%m-%d")
    csv_file = open(f"{current_date}.csv", "a+", newline="")
    
    csv_writer = csv.writer(csv_file)
    

    # Function to run face recognition
    def run_face_recognition():
        video_capture = cv2.VideoCapture(0)
        s = True

        existing_names = set(row[0] for row in csv.reader(csv_file))  # Collect existing names from the CSV file   
        

        while s:
            _, frame = video_capture.read()
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = small_frame[:, :, ::-1]
            
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(small_frame, face_locations)
            face_names = []

            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = ""
                face_distance = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distance)
                if matches[best_match_index]:
                    name = known_faces_names[best_match_index]

                face_names.append(name)

               
                for name in face_names:
                    if name in known_faces_names and name in students and name not in existing_names:
                        students.remove(name)
                        print(students)
                        print(f"Attendance recorded for {name}")
                        current_time = now.strftime("%H-%M-%S")
                        csv_writer.writerow([name, current_time, "Present"])
                        existing_names.add(name)  # Add the name to the set of existing names
                        
                        s = False  # Set s to False to exit the loop after recording attendance
                        break  # Break the loop once attendance has been recorded for a name

            cv2.imshow("Attendance System", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        video_capture.release()
        cv2.destroyAllWindows()
        csv_file.close()

    # Call the function to run face recognition
    run_face_recognition()

    return redirect(url_for('show_table'))

@app.route('/table')
def show_table():
    # Get the current date
    current_date = datetime.now().strftime("%Y-%m-%d")
    # Read the CSV file to get attendance data
    attendance=[]
    try:
        with open(f"{current_date}.csv", newline="") as csv_file:
            csv_reader = csv.reader(csv_file)
            attendance = list(csv_reader)
    except FileNotFoundError:
        pass
    # Render the table.html template and pass the attendance data
    return render_template('attendance.html', attendance=attendance)

@app.route('/')
def index():
    return render_template('index.html')


if __name__ == '__main__':
    # Start Flask application
    app.run(debug=True)