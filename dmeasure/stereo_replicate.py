import numpy as np
import cv2
from openpyxl import Workbook # Used for writing data into an Excel file
from sklearn.preprocessing import normalize


from dmeasure import utils


left_stereo_map, right_stereo_map = utils.distortion_calibration()
stereo, stereo_left, stereo_right  = utils.stereo_builder()
wls_filter = utils.wls_filter_builder(stereo)


# Call the two cameras
cam_right = cv2.VideoCapture(0)   # Wenn 0 then Right Cam and wenn 2 Left Cam
cam_left = cv2.VideoCapture(2)

while True:
    # Start Reading Camera images
    retR, frame_right = cam_right.read()
    retL, frame_left  = cam_left.read()

    filt_color = utils.preprocess_video_frame(frame_left, frame_right, 
                                              left_stereo_map, right_stereo_map, 
                                              stereo_left, stereo_right, 
                                              wls_filter)

    cv2.imshow('Filtered Color Depth', filt_color)

    # Mouse click
    cv2.setMouseCallback("Filtered Color Depth",utils.coords_mouse_disp, filt_color)
    
    # End the Programme
    if cv2.waitKey(1) & 0xFF == ord(' '):
        break
    
# Save excel
##wb.save("data4.xlsx")

# Release the Cameras
CamR.release()
CamL.release()
cv2.destroyAllWindows()
