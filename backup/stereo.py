import serial # you need to install the pySerial :pyserial.sourceforge.net
import cv2
import numpy as np
import imutils
import time

def nothing(x):
    pass

arduino = serial.Serial("COM10", 115200, timeout=1)
#serial.Serial("COM10", 115200, timeout=1)
#arduino = serial.Serial('/dev/ttyUSB0', 115200)
# arduino=serial.Serial('/dev/ttyACM0',115200)
# serial.Serial('/dev/ttyUSB0')


kernelOpenR=np.ones((5,5))
kernelOpenL=np.ones((5,5))
kernelCloseR=np.ones((20,20))
kernelCloseL=np.ones((20,20))

font = cv2.FONT_HERSHEY_SIMPLEX
fontscale = 2
fontcolor = (255,255,255)


cv2.namedWindow("Kalibrasi")

cv2.namedWindow("Kalibrasi")

cv2.createTrackbar("L - H", "Kalibrasi", 0, 179, nothing)
cv2.createTrackbar("L - S", "Kalibrasi", 135, 255, nothing)
cv2.createTrackbar("L - V", "Kalibrasi", 185, 255, nothing)
cv2.createTrackbar("U - H", "Kalibrasi", 179, 179, nothing)
cv2.createTrackbar("U - S", "Kalibrasi", 255, 255, nothing)
cv2.createTrackbar("U - V", "Kalibrasi", 255, 255, nothing)

FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FRAME_WIDTH2 = 640
FRAME_HEIGHT2 = 480

cap = cv2.VideoCapture(1)
cap2 = cv2.VideoCapture(2)
# Set properties. Each returns === True on success (i.e. correct resolution)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH) #640//640
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT) #480//480
cap2.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH) #640//640
cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT) #480//480

# Create old frame
_, frame = cap.read()
_, frame1 = cap2.read()


while True:

    try:    
        ret,imgR = cap.read()
        ret, imgL = cap2.read()
        
        
        hsvR = cv2.cvtColor(imgR, cv2.COLOR_BGR2HSV)
        hsvL = cv2.cvtColor(imgL, cv2.COLOR_BGR2HSV)
        
        l_h = cv2.getTrackbarPos("L - H", "Kalibrasi")
        l_s = cv2.getTrackbarPos("L - S", "Kalibrasi")
        l_v = cv2.getTrackbarPos("L - V", "Kalibrasi")
        u_h = cv2.getTrackbarPos("U - H", "Kalibrasi")
        u_s = cv2.getTrackbarPos("U - S", "Kalibrasi")
        u_v = cv2.getTrackbarPos("U - V", "Kalibrasi")
        lower_value = np.array([l_h, l_s, l_v])
        upper_value = np.array([u_h, u_s, u_v])
        
        maskR = cv2.inRange(hsvR, lower_value, upper_value)
        maskL = cv2.inRange(hsvL, lower_value, upper_value)
        
        maskOpenR=cv2.morphologyEx(maskR,cv2.MORPH_OPEN,kernelOpenR)
        maskOpenL=cv2.morphologyEx(maskL,cv2.MORPH_OPEN,kernelOpenL)
        maskCloseR=cv2.morphologyEx(maskOpenR,cv2.MORPH_CLOSE,kernelCloseR)
        maskCloseL=cv2.morphologyEx(maskOpenL,cv2.MORPH_CLOSE,kernelCloseL)
        
        maskFinalR=maskCloseR
        maskFinalL=maskCloseL
        contsR=cv2.findContours(maskFinalR.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]
        contsL=cv2.findContours(maskFinalL.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]
        
        centerR = None
        centerL = None
        
        # x1=0
        # y1=0
        # x2=0
        # y2=0
        
        
        for iR in range(len(contsR)):
            if len(contsR) > 0:
                c = max(contsR, key=cv2.contourArea)
                ((x1, y1), radius) = cv2.minEnclosingCircle(c)
                M = cv2.moments(c)
            
                if(M["m00"]==0): 
                    shape = "line" 
                else:
                    center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                    cv2.circle(imgR, (int(x1),int(y1)),int(radius), (0,255,0), 2)
                    cv2.putText(imgR, 'center: {}, {}'.format(int(x1), int(y1)), (int(x1-radius),int(y1-radius)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
                    
                    
                    #cv2.line(imgR,(0,0),(200,640),(255,0,0),5)
                    
                    # #rumus pake satu titik lurus
                    # mgariskanan = ((640 - 0)/(200-0)) 
                    # xgariskanan = ((y1-640)+(mgariskanan*200))/ mgariskanan
                    # errorx1= x1-xgariskanan
                    
                    
                    
                    cv2.line(imgR,(226,216),(159,132),(255,0,0),5)
                    cv2.line(imgR,(398,217),(261,139),(255,0,0),5)
                    
                    # rumus pake 2 gradien
                       # mgaris = y-y1 / x-x
                        
                       
             
                             
                    
                    #pid 
                    #cv2.putText(imgR, 'ss {}, {}'.format(xgariskanan,errorx1) , 0.6, (0,0,255), 2)
                    # print(xgariskanan,errorx1)
          
        for iL in range(len(contsL)):
            if len(contsL) > 0:
                c = max(contsL, key=cv2.contourArea)
                ((x2, y2), radius) = cv2.minEnclosingCircle(c)
                M = cv2.moments(c)
            
                if(M["m00"]==0): 
                    shape = "line" 
                else:
                    center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                    cv2.circle(imgL, (int(x2),int(y2)),int(radius), (0,255,0), 2)
                    cv2.putText(imgL, 'center: {}, {}'.format(int(x2), int(y2)), (int(x2-radius),int(y2-radius)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
            
                    cv2.line(imgL,(393,216),(476,127),(255,0,0),5)
                    cv2.line(imgL,(221,217),(362,145),(255,0,0),5)
                    
                    # rumus pake 2 gradien
                       
            
            
            
            
            
            
            
            #cv2.line(imgL,(640,0),(120,640),(255,0,0),5)
            
             
             
     #       print (x1,y1,x2,y2)
            
            
                 
        
     #   cv2.imshow("Mask Close Kanan",maskCloseR)
      #  cv2.imshow("Mask Open Kanan",maskOpenR)
       # cv2.imshow("Mask Kanan",maskR)
        cv2.imshow("KameraKanan",imgR)
        
        #cv2.imshow("Mask Close Kiri",maskCloseL)
        #cv2.imshow("Mask Open Kiri",maskOpenL)
        cv2.imshow("Mask Kiri",maskL)
        cv2.imshow("KameraKiri",imgL)
        
        
        
        mgariskanan1=((216 - 132)/(226-159))
        mgariskanan2 = ((217 - 139)/(398-261))
                    
                        # milih salah siji
        xgariskanan1 =  ((y1-216)+(mgariskanan1*226))/ mgariskanan1
        xgariskanan2 =  ((y1-217)+(mgariskanan2*398))/ mgariskanan2
                       
        if (x1<xgariskanan1):
            flagkanan = 1
        elif (x1>=xgariskanan1) and (x1<=xgariskanan2):
            flagkanan = 2
        else:
            flagkanan = 3

        mgariskiri1 = ((216 - 127)/(294-377))
        mgariskiri2 = ((217 - 145)/(122-262))
        xgariskiri1 =  ((y1-216)+(mgariskiri1*294))/ mgariskiri1
        xgariskiri2 =  ((y1-217)+(mgariskiri2*122))/ mgariskiri2
     
        if (x2<xgariskiri1):
            flagkiri = 1
        elif (x2>=xgariskiri1) and (x2<=xgariskiri2):
            flagkiri = 2
        else: 
            flagkiri = 3
     
    

        if (x1>0) and (y1>0):
            flag = flagkanan
        elif (x2>0) and (y2>0):
            flag = flagkiri
        else:
            flag=0
          
        #if (y1<200) or (y2 <200):  #200 data jarak 3meter
         #   zonamaju = 0
        #else:
        #    zonahitung = 1
        
        
            #perhitungan jarak
        #f = (4/3.6)*640
        z = abs((8.5 * 820) / (x1-x2))
        jarak = z+((0.0014*z*z)-( 0.1149*z)-0.0237)
        print (jarak)
        #print (x1,y1,x2,y2)
        
        dataKirim = "#%d,%d,%d,%d,;;**" % (flag,jarak,y1,y2) 
        arduino.write(dataKirim.encode())
        #time.sleep(0.05)
    except:
        pass
        
        #cv2.waitKey(10)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
cv2.destroyAllWindows()
cap.release()