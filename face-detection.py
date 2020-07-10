# viola and jones methods
import cv2

faceCascade = cv2.CascadeClassifier("Resources/haarcascade_frontalface_default.xml")

vidCam = cv2.VideoCapture(0)
vidCam.set(3,640) #weight
vidCam.set(4,480) #height
vidCam.set(10,100)#brightness

while True:
    success, img = vidCam.read()
    imgGrey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(imgGrey, 1.1, 4)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cv2.imshow("video",img)
    if cv2.waitKey(1) & 0xFF ==ord("q"):
        break





