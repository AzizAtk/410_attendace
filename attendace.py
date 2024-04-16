import face_recognition
import cv2
import numpy as np
import csv
import os
from datetime import datetime

# Initialize the webcam
video_capture = cv2.VideoCapture(0)

# Get the list of student folder names in the current directory
student_folders = [f for f in os.listdir('.') if os.path.isdir(f)]
known_face_encodings = []
known_faces_names = []

# Load all images from each student folder and create encodings
for student in student_folders:
    images_path = os.path.join(student)
    images_files = [f for f in os.listdir(images_path) if f.endswith('.png')]
    encodings = []

    for image_file in images_files:
        image_path = os.path.join(images_path, image_file)
        student_image = face_recognition.load_image_file(image_path)
        student_encodings = face_recognition.face_encodings(student_image)

        if student_encodings:
            encodings.append(student_encodings[0])

    if encodings:
        # Average the encodings to get a single representation
        average_encoding = np.mean(encodings, axis=0)
        known_face_encodings.append(average_encoding)
        known_faces_names.append(student)
    else:
        print(f"No valid face encodings found in {images_path}")

students = known_faces_names.copy()

face_locations = []
face_encodings = []
face_names = []
s = True

now = datetime.now()
current_date = now.strftime("%Y-%m-%d")

f = open(current_date + '.csv', 'w+', newline='')
lnwriter = csv.writer(f)

while True:
    _, frame = video_capture.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]
    now = datetime.now()
    frame_skip = 5  # Only process every 5th frame
    count = 0

    if s:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        face_landmarks_list = face_recognition.face_landmarks(rgb_small_frame)
        face_names = []

        for face_location, face_encoding, face_landmarks in zip(face_locations, face_encodings, face_landmarks_list):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = ""
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)

            if matches[best_match_index]:
                name = known_faces_names[best_match_index]
                accuracy = (1 - face_distances[best_match_index]) * 100  # Accuracy as a percentage

            face_names.append(name)
            if name:
                # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                top, right, bottom, left = [x * 4 for x in face_location]

                # Draw a box around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

                # Highlighting the facial features
                # Highlighting the facial features
                #for facial_feature in ['left_eye', 'right_eye', 'top_lip', 'bottom_lip', 'nose_bridge', 'nose_tip']:
                    #points = [(x * 4, y * 4) for (x, y) in face_landmarks[facial_feature]]
                    #cv2.polylines(frame, [np.array(points, dtype=np.int32)], isClosed=False, color=(255, 255, 0), thickness=1)


                # Draw label with a name and accuracy below the face
                #cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                text = f"{name} ({accuracy:.2f}%)"
                cv2.putText(frame, text, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

                if name in students:
                    students.remove(name)
                    current_time = now.strftime("%H-%M-%S")
                    lnwriter.writerow([name, current_time])

    cv2.imshow("attendance system", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
f.close()
