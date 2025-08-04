from sklearn.neighbors import KNeighborsClassifier

import cv2
import pickle
import numpy as np
import os

video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')


names_path = 'data/names.pkl'  
faces_path = 'data/face_data.pkl'

with open(names_path, 'rb') as f:
    LABLES = pickle.load(f)
with open(faces_path, 'rb') as f:
    FACES = pickle.load(f)


knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES, LABLES)



cole = str('mewhentacobell')

while True:
    ret,frame=video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        crop_img = frame[y:y + h, x:x + w, : ]  
        resized_img = cv2.resize(crop_img, (50, 50,)).flatten().reshape(1, -1)
        output = knn.predict(resized_img)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1 )
        cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 2 )
        cv2.rectangle(frame, (x, y-40), (x+w,y), (50, 50, 255), -1 )
        cv2.putText(frame, str(output[0]), (x, y - 15), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cv2.imshow("Frame", frame)
    k = cv2.waitKey(1)
    if k == ord('q'):
        break

video.release()

cv2.destroyAllWindows()

