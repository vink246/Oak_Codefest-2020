
import cv2
import numpy as np
import arduino

#https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
face_cascade = cv2.CascadeClassifier('/home/samarth/Documents/opencv-master/data/haarcascades/haarcascade_frontalface_default.xml')
#https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_eye.xml
eye_cascade = cv2.CascadeClassifier('/home/samarth/Documents/opencv-master/data/haarcascades/haarcascade_eye.xml')
#haarcascade_eye_tree_eyeglasses.xml
glass_cascade = cv2.CascadeClassifier('/home/samarth/Documents/opencv-master/data/haarcascades/haarcascade_eye_tree_eyeglasses.xml')
#haarcascade_frontalcatface.xml
face_cascade2 = cv2.CascadeClassifier('/home/samarth/Documents/opencv-master/data/haarcascades/haarcascade_frontalcatface_extended.xml')
#
face_cascadeAlt = cv2.CascadeClassifier('/home/samarth/Documents/opencv-master/data/haarcascades/haarcascade_frontalface_alt.xml')
#
face_cascadeAltTree = cv2.CascadeClassifier('/home/samarth/Documents/opencv-master/data/haarcascades/haarcascade_frontalface_alt_tree.xml')
#
face_cascadeAlt2 = cv2.CascadeClassifier('/home/samarth/Documents/opencv-master/data/haarcascades/haarcascade_frontalface_alt2.xml')
#
body_cascade = cv2.CascadeClassifier('/home/samarth/Documents/opencv-master/data/haarcascades/haarcascade_upperbody.xml')
#
smile_cascade = cv2.CascadeClassifier('/home/samarth/Documents/opencv-master/data/haarcascades/haarcascade_smile.xml')

#defining functions for nn
def get_output_layers(net):
    
    layer_names = net.getLayerNames()
    
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers


def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):

    label = str(classes[class_id])

    color = COLORS[class_id]

    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)

    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

#mapping function
def translate(value, leftMin, leftMax, rightMin, rightMax):
    # Figure out how 'wide' each range is
    leftSpan = leftMax - leftMin
    rightSpan = rightMax - rightMin

    # Convert the left range into a 0-1 range (float)
    valueScaled = float(value - leftMin) / float(leftSpan)

    # Convert the 0-1 range into a value in the right range.
    return rightMin + (valueScaled * rightSpan)

classes = None
#path of class names (.txt)
with open('/home/samarth/Documents/PythonRelated/yolov3.txt', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

COLORS = np.random.uniform(0, 255, size=(len(classes), 3))
#path of weights (.weights) and config file (.cfg)
net = cv2.dnn.readNet('/home/samarth/Documents/PythonRelated/yolov3.weights', '/home/samarth/Documents/PythonRelated/yolov3.cfg')
#path of input footage
cap = cv2.VideoCapture('/home/samarth/Documents/Inputs/testVid.webm')
#'DIVX' for Windows
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
ret, image = cap.read()
#cropping input frame size
crop_img = image[0:336,80:556]
#path and dimensions of output footage
writeOut = cv2.VideoWriter('/home/samarth/Documents/Outputs/output.mp4',fourcc, 20.0,(int(crop_img.shape[1]),int(crop_img.shape[0])))
#----------------------------------------------------------
#defining constants
WEIGHT_EYEDEV = 0.9
WEIGHT_OBJECT = 4.5
WEIGHT_BODYDEV = 0.4
#defining variables
tempE = 0
tempB = 0
tempED = 0
tempBD = 0
tempDis = 0
deviationEye = []
deviationBody = []
badobj = []
FRAMEGAP = 11
frameSetC = 0
frameC = 0
finalEyeDev = 0
finalBodyDev = 0
isObjectDetected = 'not detected'
obDec = 1
finalDisVal = 0
#---------------
while(cap.isOpened()):
    ret, image = cap.read()
    if ret==True:
        #resetting variables and lists
        frameC+=1
        eyesX = []
        eyesY = []
        bodyX = 0
        bodyY = 0
        #cropping input frames for processing
        image = image[0:336,80:556]

        Width = image.shape[1]
        Height = image.shape[0]
        scale = 0.00392
        blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)

        net.setInput(blob)
        #finding nn outputs
        outs = net.forward(get_output_layers(net))

        class_ids = []
        confidences = []
        boxes = []
        conf_threshold = 0.5
        nms_threshold = 0.4


        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    #setting detected coordinates
                    center_x = int(detection[0] * Width)
                    center_y = int(detection[1] * Height)
                    w = int(detection[2] * Width)
                    h = int(detection[3] * Height)
                    x = center_x - w / 2
                    y = center_y - h / 2
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])

        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
        #drawing bounding boxes for body and foreign objects
        for i in indices:
            i = i[0]
            box = boxes[i]
            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]
            draw_prediction(image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))
            #printing out the detected classes
            print(str(classes[class_ids[i]]))
            #check if foreign objects are present
            if str(classes[class_ids[i]]) == 'cell phone' or str(classes[class_ids[i]]) == 'cup' or str(classes[class_ids[i]]) == 'laptop':
                badobj.append(1)
            #obtain coordinates for tracking body
            if str(classes[class_ids[i]]) == 'person' and w > 50 and h > 50:
                bodyX = round(x+w)
                bodyY = round(y+h)
        #Setting up for Haar-Cascades 
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        faces2 = face_cascadeAlt.detectMultiScale(gray, 1.3, 5)
        #face detection 2
        for (x,y,w,h) in faces2:
            cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = image[y:y+h, x:x+w]
            #eye detection
            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex,ey,ew,eh) in eyes[:2]:
                cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
                eyesX.append(ex+x)
                eyesY.append(ey+y)
            #smiles = smile_cascade.detectMultiScale(roi_gray)
            #for (ex,ey,ew,eh) in smiles:
                #cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

        #face detection 3
        faces3 = face_cascadeAltTree.detectMultiScale(gray, 1.3, 5)

        for (x,y,w,h) in faces3:
            cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = image[y:y+h, x:x+w]

            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex,ey,ew,eh) in eyes[:2]:
                cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
                eyesX.append(ex+x)
                eyesY.append(ey+y)
            #smiles = smile_cascade.detectMultiScale(roi_gray)
            #for (ex,ey,ew,eh) in smiles:
                #cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

        #face detection 4
        faces4 = face_cascadeAlt2.detectMultiScale(gray, 1.3, 5)

        for (x,y,w,h) in faces4:
            cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = image[y:y+h, x:x+w]

            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex,ey,ew,eh) in eyes[:2]:
                cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
                eyesX.append(ex+x)
                eyesY.append(ey+y)
            #smiles = smile_cascade.detectMultiScale(roi_gray)
            #for (ex,ey,ew,eh) in smiles[:1]:
                #cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

        #potential body detection
        #body = body_cascade.detectMultiScale(gray, 1.3, 5)

        #for (x,y,w,h) in body:
            #cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
            #roi_gray = gray[y:y+h, x:x+w]
            #roi_color = image[y:y+h, x:x+w]

        #smiles = smile_cascade.detectMultiScale(gray, 1.3, 5)

        #for (x,y,w,h) in smiles:
            #cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
            #roi_gray = gray[y:y+h, x:x+w]
            #roi_color = image[y:y+h, x:x+w]

        #glasses = glass_cascade.detectMultiScale(roi_gray)
        #for (ex,ey,ew,eh) in glasses:
            #cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

        #face detection 1
        for (x,y,w,h) in faces:
            cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = image[y:y+h, x:x+w]
   
            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex,ey,ew,eh) in eyes[:2]:
                cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
                eyesX.append(ex+x)
                eyesY.append(ey+y)
        #------
        sumEX = 0
        sumEY = 0
        #calculating average of all eye bounding box positions
        for xc in eyesX:
            sumEX += xc
        for yc in eyesY:
            sumEY += yc
        #calculating eye deviation
        if len(eyesX)>0:
            avgEX = sumEX/len(eyesX)
            avgEY = sumEY/len(eyesY)

            avgEX = int(avgEX)
            avgEY = int(avgEY)
        
            devE = pow((pow(avgEX,2)+pow(avgEY,2)),0.5)
            deviationEye.append(abs(devE - tempE))
            tempE = devE
            #draw circle to mark mean position
            image = cv2.circle(image,(avgEX,avgEY), 5, (0,0,255), -1)

        #calculating body deviation
        devB = pow((pow(bodyX,2)+pow(bodyY,2)),0.5)
        deviationBody.append(abs(devB - tempB))
        tempB = devB
        #draw circle to mark mean position
        image = cv2.circle(image,(bodyX,bodyY), 5, (255,0,0), -1)

        #calculating mean eye deviation and foreign object detection
        if frameC > FRAMEGAP-1:
            frameSetC += 1
            frameC = 0
            if len(deviationEye) > 1:
                deviationEye.pop(0)
                totalDevE = 0
                for val in deviationEye:
                    totalDevE += val
                finalEyeDev = totalDevE/len(deviationEye)
                if frameSetC > 0:
                    finalEyeDev = (finalEyeDev+tempED)/2
                totalobj = 0
                for obj in badobj:
                    totalobj += obj
                meanobj = 0
                if len(badobj) > 0:
                    meanobj = totalobj/len(badobj)
                isObjectDetected = 'not detected'
                obDec = 1
                if meanobj > 0.5:
                    isObjectDetected = 'detected'
                    obDec = 2
                deviationEye = []
                badobj = []
                tempE = 0
                tempED = finalEyeDev
                #printing out mean eye deviation
                print('mean eye deviation: '+str(finalEyeDev))

            #calculating mean body deviation
            if len(deviationBody) > 1:
                deviationBody.pop(0)
                totalDevB = 0
                for val in deviationBody:
                    totalDevB += val
                finalBodyDev = totalDevB/len(deviationBody)
                if frameSetC > 0:
                    finalBodyDev = (finalBodyDev+tempBD)/2
                deviationBody = []
                tempB = 0
                tempBD = finalBodyDev
                print('mean body deviation: '+str(finalBodyDev))

            #calculating driver distraction level
            finalDisVal = (finalEyeDev*WEIGHT_EYEDEV)+(finalBodyDev*WEIGHT_BODYDEV)+(obDec*WEIGHT_OBJECT)
            #if frameSetC > 0:
            #    finalDisVal = (finalDisVal+tempDis)/2
            finalDisVal = translate(finalDisVal, 2, 17, 0, 10)
            #calculating rating
            rating = arduino.main(finalDisVal)
            print("rating: " + str(rating))
            #printing out distracted level
            print('distraction level: '+str(finalDisVal))

        #printing out no. of frame sets which have passed
        print(frameSetC)
        #printing out list pertaining to foreign objects
        print(badobj)
        #print out to frame
        cv2.putText(image,str(finalEyeDev),(10,220), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,0,0),2,cv2.LINE_AA)
        cv2.putText(image,isObjectDetected,(10,250), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,0,0),2,cv2.LINE_AA)
        cv2.putText(image,str(finalBodyDev),(10,280), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,0,0),2,cv2.LINE_AA)
        cv2.putText(image,"distraction level: "+str(finalDisVal),(10,310), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,0,0),2,cv2.LINE_AA)
        #write out the edited frame so that it can be saved
        writeOut.write(image)
        #show edited frame
        cv2.imshow('img',image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
#finishing up computer vision processes
cap.release()
writeOut.release()
cv2.destroyAllWindows()
