
import numpy as np
import cv2
from openpyxl import Workbook # Used for writing data into an Excel file
from sklearn.preprocessing import normalize


from dmeasure import utils



cam_left = cv2.VideoCapture(4)
cam_right = cv2.VideoCapture(0)   # Wenn 0 then Right Cam and wenn 2 Left Cam


while True:
    # Start Reading Camera images
    retL, frame_left  = cam_left.read()
    retR, frame_right = cam_right.read()

    cv2.imshow('Frame Left', frame_left)
    cv2.imshow('Frame Right', frame_right)
    q
    # End the Programme
    if cv2.waitKey(1) & 0xFF == ord(' '):
        break
    
# Save excel
##wb.save("data4.xlsx")

# Release the Cameras
cam_right.release()
cam_left.release()
cv2.destroyAllWindows()