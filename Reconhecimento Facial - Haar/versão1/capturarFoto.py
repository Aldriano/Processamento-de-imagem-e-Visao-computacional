#https://www.imed.edu.br/Uploads/JEAN%20CARLOS%20MAFFI.pdf
#https://www.youtube.com/watch?v=zqWmMY6buFQ
import cv2              
import util
import numpy as np          
import sys
import math
import time

WHITE = [255, 255, 255]

face_cas = cv2.CascadeClassifier('cascades/haarcascade_frontalcatface.xml')
eye_cascade = cv2.CascadeClassifier('cascades/haarcascade_eye.xml')
glass_cas   = cv2.CascadeClassifier('Haar/haarcascade_eye_tree_eyeglasses.xml')

ID = util.AddName()
Count = 0
cap = cv2.VideoCapture(0)                                                                           # Camera object

while Count < 50:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert the Camera to graySe
    #print("Media = ",np.average(gray))
    if np.average(gray) > 100:                                                                      # Testing the brightness of the image/ brilho da imagem
        #faces = face_cas.detectMultiScale(gray, 1.3, 5)                                                 # Detect the faces and store the positions/ Detecta o roso e guarda as posições
            
        faces = face_cas.detectMultiScale(
            gray,     
            scaleFactor=1.2,
            minNeighbors=5,     
            minSize=(20, 20)
        )
        if len(faces) == 0:
            print("No faces found..")
        else:
            #print faces
            #print faces.shape
            #print "Number of faces detected: " + str(faces.shape[0])
            for (x, y, w, h) in faces:                                                                  # Frames  LOCATION X, Y  WIDTH, HEIGHT
                cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
                #roi_gray = gray[y:y+h, x:x+w]
            
                FaceImage = gray[y - int(h / 2): y + int(h * 1.5), x - int(x / 2): x + int(w * 1.5)]    # The Face is isolated and cropped/ O rosto é isolado e cortado
                Img = (util.DetectEyes(FaceImage))
                #cv2.putText(gray, "FACE DETECTED", (x+(w/2), y-5), cv2.FONT_HERSHEY_DUPLEX, .4, WHITE, 2, cv2.LINE_AA)
                if Img is not None:
                    print("não achou os olhos")
                    frame = Img                                                                         # Show the detected faces
                else:
                    frame = gray[y: y+h, x: x+w]
                    print("Img existe")
                cv2.imwrite("fotos/User." + str(ID) + "." + str(Count) + ".jpg", frame)
                cv2.waitKey(300)
                cv2.imshow("CAPTURED PHOTO", frame)                                                     # show the captured image
                Count = Count + 1
                
    cv2.imshow('Face Recognition System Capture Faces', gray)                                       # Show the video
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
print("FACE CAPTURE FOR THE SUBJECT IS COMPLETE")
cap.release()
cv2.destroyAllWindows()
