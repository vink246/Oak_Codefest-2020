import numpy as np
import cv2

# multiple cascades: https://github.com/Itseez/opencv/tree/master/data/haarcascades

#https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
face_cascade = cv2.CascadeClassifier('C:\\Users\\Vineet\\Documents\\opencv\\build\\etc\\haarcascades\\haarcascade_frontalface_default.xml')
#https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_eye.xml
eye_cascade = cv2.CascadeClassifier('C:\\Users\\Vineet\\Documents\\opencv\\build\\etc\\haarcascades\\haarcascade_eye.xml')
#haarcascade_eye_tree_eyeglasses.xml
glass_cascade = cv2.CascadeClassifier('C:\\Users\\Vineet\\Documents\\opencv\\build\\etc\\haarcascades\\haarcascade_eye_tree_eyeglasses.xml')
#haarcascade_frontalcatface.xml
face_cascade2 = cv2.CascadeClassifier('C:\\Users\\Vineet\\Documents\\opencv\\build\\etc\\haarcascades\\haarcascade_frontalcatface_extended.xml')
#
face_cascadeAlt = cv2.CascadeClassifier('C:\\Users\\Vineet\\Documents\\opencv\\build\\etc\\haarcascades\\haarcascade_frontalface_alt.xml')
#
face_cascadeAltTree = cv2.CascadeClassifier('C:\\Users\\Vineet\\Documents\\opencv\\build\\etc\\haarcascades\\haarcascade_frontalface_alt_tree.xml')
#
face_cascadeAlt2 = cv2.CascadeClassifier('C:\\Users\\Vineet\\Documents\\opencv\\build\\etc\\haarcascades\\haarcascade_frontalface_alt2.xml')
#
body_cascade = cv2.CascadeClassifier('C:\\Users\\Vineet\\Documents\\opencv\\build\\etc\\haarcascades\\haarcascade_upperbody.xml')
#
smile_cascade = cv2.CascadeClassifier('C:\\Users\\Vineet\\Documents\\opencv\\build\\etc\\haarcascades\\haarcascade_smile.xml')

cap = cv2.VideoCapture(0)

while 1:
    ret, img = cap.read()
    if ret==True:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        faces2 = face_cascadeAlt.detectMultiScale(gray, 1.3, 5)

        for (x,y,w,h) in faces2:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]

            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex,ey,ew,eh) in eyes[:2]:
                cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

            #smiles = smile_cascade.detectMultiScale(roi_gray)
            #for (ex,ey,ew,eh) in smiles:
                #cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        faces3 = face_cascadeAltTree.detectMultiScale(gray, 1.3, 5)

        for (x,y,w,h) in faces3:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]

            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex,ey,ew,eh) in eyes[:2]:
                cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

            #smiles = smile_cascade.detectMultiScale(roi_gray)
            #for (ex,ey,ew,eh) in smiles:
                #cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        faces4 = face_cascadeAlt2.detectMultiScale(gray, 1.3, 5)

        for (x,y,w,h) in faces4:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]

            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex,ey,ew,eh) in eyes[:2]:
                cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

            #smiles = smile_cascade.detectMultiScale(roi_gray)
            #for (ex,ey,ew,eh) in smiles[:1]:
                #cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        #body = body_cascade.detectMultiScale(gray, 1.3, 5)

        #for (x,y,w,h) in body:
            #cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            #roi_gray = gray[y:y+h, x:x+w]
            #roi_color = img[y:y+h, x:x+w]

        #smiles = smile_cascade.detectMultiScale(gray, 1.3, 5)

        #for (x,y,w,h) in smiles:
            #cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            #roi_gray = gray[y:y+h, x:x+w]
            #roi_color = img[y:y+h, x:x+w]

        #glasses = glass_cascade.detectMultiScale(roi_gray)
        #for (ex,ey,ew,eh) in glasses:
            #cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        for (x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]

        
                
            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex,ey,ew,eh) in eyes[:2]:
                cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        
        
        cv2.imshow('frame',img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()
