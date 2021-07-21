import cv2
import numpy as np
import vehicles
import time
cnt_down=0
cap=cv2.VideoCapture("VIDEO.mp4")

#Get width and height of video
w=cap.get(3)
h=cap.get(4)
frameArea=h*w
areaTH=frameArea/400

#Lines
line_up=int(2*(h/5))
line_down=int(3*(h/5))
up_limit=int(1*(h/5))
down_limit=int(4*(h/5))
line_down_color=(255,0,0)
line_up_color=(255,0,255)
pt1 =  [0, line_down]
pt2 =  [w, line_down]
pts_L1 = np.array([pt1,pt2], np.int32)
pts_L1 = pts_L1.reshape((-1,1,2))
pt3 =  [0, line_up]
pt4 =  [w, line_up]
pts_L2 = np.array([pt3,pt4], np.int32)
pts_L2 = pts_L2.reshape((-1,1,2))
pt5 =  [0, up_limit]
pt6 =  [w, up_limit]
pts_L3 = np.array([pt5,pt6], np.int32)
pts_L3 = pts_L3.reshape((-1,1,2))
pt7 =  [0, down_limit]
pt8 =  [w, down_limit]
pts_L4 = np.array([pt7,pt8], np.int32)
pts_L4 = pts_L4.reshape((-1,1,2))

#Background Subtractor
fgbg=cv2.createBackgroundSubtractorMOG2(detectShadows=True)

#Kernals
kernalOp = np.ones((3,3),np.uint8)
kernalOp2 = np.ones((5,5),np.uint8)
kernalCl = np.ones((11,11),np.uint8)
font = cv2.FONT_HERSHEY_SIMPLEX
cars = []
max_p_age = 5
pid = 1
while(cap.isOpened()):
    ret,frame=cap.read()
    for i in cars:
        i.age_one()
    fgmask=fgbg.apply(frame)
    fgmask2=fgbg.apply(frame)
    if ret==True:
        #Binarization
        ret,imBin=cv2.threshold(fgmask,200,255,cv2.THRESH_BINARY)
        ret,imBin2=cv2.threshold(fgmask2,200,255,cv2.THRESH_BINARY)
        #OPening i.e First Erode the dilate
        mask=cv2.morphologyEx(imBin,cv2.MORPH_OPEN,kernalOp)
        mask2=cv2.morphologyEx(imBin2,cv2.MORPH_CLOSE,kernalOp)
        #Closing i.e First Dilate then Erode
        mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernalCl)
        mask2=cv2.morphologyEx(mask2,cv2.MORPH_CLOSE,kernalCl)
        #Find Contours
        countours0,hierarchy=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        for cnt in countours0:
            area=cv2.contourArea(cnt)
            if area>areaTH:
                ####Tracking######
                m=cv2.moments(cnt)
                cx=int(m['m10']/m['m00'])
                cy=int(m['m01']/m['m00'])
                x,y,w,h=cv2.boundingRect(cnt)
                new=True
                if cy in range(up_limit,down_limit):
                    for i in cars:
                        if abs(x - i.getX()) <= w and abs(y - i.getY()) <= h:
                            new = False
                            i.updateCoords(cx, cy)
                            if i.going_DOWN(line_down,line_up)==True:
                                cnt_down+=1
                            break
                        if i.getState()=='1':
                            if i.getDir()=='down'and i.getY()>down_limit:
                                i.setDone()
                            elif i.getDir()=='up'and i.getY()<up_limit:
                                i.setDone()
                        if i.timedOut():
                            index=cars.index(i)
                            cars.pop(index)
                            del i
                    if new==True: #If nothing is detected,create new
                        p=vehicles.Car(pid,cx,cy,max_p_age)
                        cars.append(p)
                        pid+=1
                img=cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        for i in cars:
            cv2.putText(frame, str(i.getId()), (i.getX(), i.getY()), font, 1, i.getRGB(), 4, cv2.LINE_AA)
        str_down= str(cnt_down)
        cv2.putText(frame, str_down, (750, 50), font, 1.5, (255, 255, 255), 6, cv2.LINE_AA)
        cv2.putText(frame, str_down, (750, 50), font, 1.5, (255, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, "Kareem Hamdi", (365, 45), font, 0.9, (255, 255, 255), 5, cv2.LINE_AA)
        cv2.putText(frame, "Kareem Hamdi", (365, 45), font, 0.9, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow('Frame',frame)
        if cv2.waitKey(1)&0xff==ord('q'):
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()









