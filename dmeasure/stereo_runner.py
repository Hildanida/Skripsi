# Package importation
import numpy as np
import cv2
from openpyxl import Workbook # Used for writing data into an Excel file
from sklearn.preprocessing import normalize


from dmeasure import utils



def run(cam_left: int, cam_right: int, chessboard_path: str):


    left_stereo_map, right_stereo_map = utils.distortion_calibration(chessboard_path)
    stereo, stereo_left, stereo_right  = utils.stereo_builder()
    wls_filter = utils.wls_filter_builder(stereo)


    # Call the two cameras
    cam_right_cap = cv2.VideoCapture(cam_right)   # Wenn 0 then Right Cam and wenn 2 Left Cam
    cam_left_cap = cv2.VideoCapture(cam_left)

    while True:
        # Start Reading Camera images
        retR, frame_right = cam_right_cap.read()
        retL, frame_left  = cam_left_cap.read()

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
    cam_right.release()
    cam_left.release()

    cv2.destroyAllWindows()

