import cv2
import numpy as np
import dlib
from imutils import face_utils
from playsound import playsound


cap = cv2.VideoCapture(0)


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

sleep = 0
drowsy = 0
active = 0
status=""
color=(0,0,0)
yawnCount=0

def compute(ptA,ptB):
    dist = np.linalg.norm(ptA - ptB)
    return dist

def blinked(a,b,c,d,e,f):
    up = compute(b,d) + compute(c,e)
    down = compute(a,f)
    ratio = up/(2.0*down)

    #Checking if it is blinked
    if(ratio>0.25):
        return 2
    elif(ratio>0.21 and ratio<=0.25):
        return 1
    else:
        return 0

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    #detected face in faces array
    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()

        face_frame = frame.copy()
        cv2.rectangle(face_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        landmarks = predictor(gray, face)
        
        #print('landmarks 1',landmarks)
        landmarks = face_utils.shape_to_np(landmarks)
        
        #print('landmarks 2',landmarks)
        #print('landmarks[1][1]',landmarks[1][1])
        
        #The numbers are actually the landmarks which will show eye
        left_blink = blinked(landmarks[36],landmarks[37], 
            landmarks[38], landmarks[41], landmarks[40], landmarks[39])
        
        right_blink = blinked(landmarks[42],landmarks[43], 
            landmarks[44], landmarks[47], landmarks[46], landmarks[45])
        
        
        ##for mouth
        x2 = landmarks[51][0]
        y2 = landmarks[51][1]
        cv2.circle(frame, (x2, y2), 2, (255, 0, 0), -1)
        
        x3 = landmarks[57][0]
        y3 = landmarks[57][1]
        cv2.circle(frame, (x3, y3), 2, (255, 0, 0), -1)
        
        

        #Now judge what to do for the eye blinks
        if(left_blink==0 or right_blink==0):
            sleep+=1
            drowsy=0
            active=0
            if(sleep>10):
                status="SLEEPING !!!"
                color = (255,0,0)
                playsound('b.mp3')

        elif(y3- y2 >28):
            print("drowsy")
            
            drowsy+=1
            print("drowsy val",drowsy)
            
            if(drowsy >20):
                    status="Drowsy !!!"
                    color = (255,0,0)
                    playsound('b.mp3')
                    

        else:
            #drowsy=0
            sleep=0
            active+=1
            if(active>6):
                drowsy = 0
                status="Active :)"
                color = (0,255,0)

        cv2.putText(frame, status, (100,100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color,3)

        for n in range(0, 68):
            (x,y) = landmarks[n]
            cv2.circle(face_frame, (x, y), 1, (255, 255, 255), -1)

        cv2.imshow("Frame", frame)
        cv2.imshow("Result of detector", face_frame)
    key = cv2.waitKey(1)
    if key == 27:
        break
