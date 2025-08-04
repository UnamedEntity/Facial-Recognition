import cv2
import pickle
import numpy as np
import os

video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')


face_data = []


i=0

name = input("Enter your name: ")
cole = str('mewhentacobell')
while True:
    ret,frame=video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        crop_img = frame[y:y + h, x:x + w, : ]  
        resized_img = cv2.resize(crop_img, (50, 50,))
        if len(face_data)<=100 and i%10==0:
            face_data.append(resized_img)
        i += 1
        cv2.putText(frame , str(len(face_data)), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cv2.imshow("Frame", frame)
    k = cv2.waitKey(1)
    if k == ord('q') or len(face_data)==100:
        break

video.release()

cv2.destroyAllWindows()

face_data = np.asarray(face_data)
face_data = face_data.reshape(100, -1)

# Ensure the data directory exists

# Save names
names_path = 'data/names.pkl'
if not os.path.exists(names_path):
    names = [name] * 100
    with open(names_path, 'wb') as f:
        pickle.dump(names, f)
else:
    with open(names_path, 'rb') as f:
        names = pickle.load(f)
    names = names + [name] * 100
    with open(names_path, 'wb') as f:
        pickle.dump(names, f)

# Save face data
faces_path = 'data/face_data.pkl'
if not os.path.exists(faces_path):
    faces = face_data
    with open(faces_path, 'wb') as f:
        pickle.dump(faces, f)
else:
    with open(faces_path, 'rb') as f:
        faces = pickle.load(f)
    faces = np.append(faces, face_data, axis=0)
    with open(faces_path, 'wb') as f:
        pickle.dump(faces, f)

    