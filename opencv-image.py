import cv2

#face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#img = cv2.imread('img/faces.jpg')

face_cascade=cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
img=cv2.imread('img/faces.jpg')
img2=cv2.imread('img/faces2.jpg')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
#faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
faces=face_cascade.detectMultiScale(gray,1.1,5,minSize=(30,30))
faces2=face_cascade.detectMultiScale(gray2,1.1,5,minSize=(30,30))

#for (x, y, w, h) in faces:
#    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

for(x,y,w,h)in faces:
    cv2.rectangle(img,(x,y),(x+w,y+w),(255,0,0),2)
for(x,y,w,h)in faces2:
    cv2.rectangle(img2,(x,y),(x+w,y+w),(255,0,0),2)


cv2.imshow('Detected Faces 1', img)
cv2.imshow('Detected Faces 2', img2)

cv2.waitKey(0)
cv2.destroyAllWindows()