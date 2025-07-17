#### Import Necessary Libraries ####

import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import pyttsx3
import time
from imutils.video import VideoStream
from tensorflow.keras.utils import img_to_array
from keras.models import load_model
import argparse
import imutils
import pickle

print("Loading face detector...")
protoPath = "face_detector/deploy.prototxt"
modelPath = "face_detector/res10_300x300_ssd_iter_140000.caffemodel"
net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

print("Loading Model...")
model = load_model("liveness.model")
le = pickle.loads(open("le.pickle", "rb").read())

print("Starting Video Stream")
v = VideoStream(src=0).start()
time.sleep(2.0)

path = 'ImagesAttendance'
images = []
personNames = []
myList = os.listdir(path)
print(myList)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    personNames.append(os.path.splitext(cl)[0])
    print(personNames)

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

def markAttendance(name):
    with open('Attendance.csv','r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')

encodeListKnown = findEncodings(images)
print('Encoding Complete')

while True:
    frame = v.read()
    frame = imutils.resize(frame, width=600)
    frame1 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                (300, 300), (104.0, 177.0, 123.0))

    net.setInput(blob)
    detections = net.forward()

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            startX = max(0, startX)
            startY = max(0, startY)
            endX = min(w, endX)
            endY = min(h, endY)

            face = frame[startY:endY, startX:endX]
            face = cv2.resize(face, (32, 32))
            face = face.astype("float") / 255.0
            face = img_to_array(face)
            face = np.expand_dims(face, axis=0)

            preds = model.predict(face)[0]
            j = np.argmax(preds)
            label = le.classes_[j]


            label = "{}: {:.4f}".format(label, preds[j])
            print(label)

            if preds[j] > 0.60 and j == 1:
                
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                _label = "Real: {:.4f}".format(preds[j])
                cv2.putText(frame, _label, (startX, startY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


                # If the face is real, check for attendance

                facesCurFrame = face_recognition.face_locations(frame1)
                encodesCurFrame = face_recognition.face_encodings(frame1,facesCurFrame)

                for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):
                    matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
                    faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
                    print(faceDis)
                    matchIndex = np.argmin(faceDis)

                    if matches[matchIndex]:
                        name = personNames[matchIndex].upper()
                        print(name)
                        y1,x2,y2,x1 = faceLoc
                        y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
                        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
                        cv2.rectangle(frame,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
                        cv2.putText(frame,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
                        markAttendance(name)
                        engine = pyttsx3.init()
                        engine.setProperty("rate", 120)
                        engine.say(name)
                        engine.say("Welcome To K L N college of Engineering")
                        engine.runAndWait()


            else:
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
                _label = "Unauthorized Identity !!! Fake/Spoofed: {:.4f}".format(preds[j])
                cv2.putText(frame, _label, (startX, startY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cv2.destroyAllWindows()
v.stop()
