import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier("//home//shuvrajeet//Documents//ATOM//python//opencv//haarcascade_frontalface_alt.xml")

cap = cv2.VideoCapture(0)

while(True):
    ret, frame = cap.read()
    gray_img = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_img,scaleFactor = 1.05,minNeighbors=5)
    for a,b,c,d in faces:
        frame = cv2.rectangle(frame,(a,b),(a+c,b+d),(0,255,0),3)


    cv2.imshow("frame",frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
