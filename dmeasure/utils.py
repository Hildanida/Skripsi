import numpy as np
import cv2
from openpyxl import Workbook # Used for writing data into an Excel file
from sklearn.preprocessing import normalize

from pathlib import Path

min_disp = 2
num_disp = 130-min_disp
kernel= np.ones((3,3),np.uint8)

def coords_mouse_disp(disp, event, x, y, flags):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        #print x,y,disp[y,x],filteredImg[y,x]
        average=0
        for u in range (-1,2):
            for v in range (-1,2):
                average += disp[y+u,x+v]
        average=average/9
        Distance= -593.97*average**(3) + 1506.8*average**(2) - 1373.1*average + 522.06
        Distance= np.around(Distance*0.01,decimals=2)
        print('Distance: '+ str(Distance)+' m')


def distortion_calibration(path):
    
    base_path = Path(path)
    # Termination criteria
    criteria =(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    criteria_stereo= (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Prepare object points
    objp = np.zeros((9*6,3), np.float32)
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

    # Arrays to store object points and image points from all images
    objpoints= []   # 3d points in real world space
    imgpointsR= []   # 2d points in image plane
    imgpointsL= []

    # Start calibration from the camera
    print('Starting calibration for the 2 cameras... ')
    # Call all saved images
    for i in range(5,77):   # Put the amount of pictures you have taken for the calibration inbetween range(0,?) wenn starting from the image number 0
        t= str(i)
        right_path = str(base_path.joinpath('chessboard-R'+t+'.png'))
        left_path = str(base_path.joinpath('chessboard-L'+t+'.png'))
        
        chess_img_right= cv2.imread(right_path,0)    # Right side
        chess_img_left= cv2.imread(left_path,0)    # Left side
        retR, corners_right = cv2.findChessboardCorners(chess_img_right,
                                                (9,6),None)  # Define the number of chees corners we are looking for
        retL, corners_left = cv2.findChessboardCorners(chess_img_left,
                                                (9,6),None)  # Left side
        if (True == retR) & (True == retL):
            objpoints.append(objp) 
            cv2.cornerSubPix(chess_img_right,corners_right,(11,11),(-1,-1),criteria)
            cv2.cornerSubPix(chess_img_left,corners_left,(11,11),(-1,-1),criteria)
            imgpointsR.append(corners_right)
            imgpointsL.append(corners_left)



    # Determine the new values for different parameters
    #   Right Side
    retR, mtxR, distR, rvecsR, tvecsR = cv2.calibrateCamera(objpoints,
                                                            imgpointsR,
                                                            chess_img_right.shape[::-1],None,None)
    hR,wR= chess_img_right.shape[:2]
    OmtxR, roiR= cv2.getOptimalNewCameraMatrix(mtxR,distR,
                                                    (wR,hR),1,(wR,hR))

    #   Left Side
    retL, mtxL, distL, rvecsL, tvecsL = cv2.calibrateCamera(objpoints,
                                                            imgpointsL,
                                                            chess_img_left.shape[::-1],None,None)
    hL,wL= chess_img_left.shape[:2]
    OmtxL, roiL= cv2.getOptimalNewCameraMatrix(mtxL,distL,(wL,hL),1,(wL,hL))

    print('Cameras Ready to use')

    #********************************************
    #***** Calibrate the Cameras for Stereo *****
    #********************************************

    # StereoCalibrate function
    #flags = 0
    #flags |= cv2.CALIB_FIX_INTRINSIC
    #flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
    #flags |= cv2.CALIB_USE_INTRINSIC_GUESS
    #flags |= cv2.CALIB_FIX_FOCAL_LENGTH
    #flags |= cv2.CALIB_FIX_ASPECT_RATIO
    #flags |= cv2.CALIB_ZERO_TANGENT_DIST
    #flags |= cv2.CALIB_RATIONAL_MODEL
    #flags |= cv2.CALIB_SAME_FOCAL_LENGTH
    #flags |= cv2.CALIB_FIX_K3
    #flags |= cv2.CALIB_FIX_K4
    #flags |= cv2.CALIB_FIX_K5
    retS, MLS, dLS, MRS, dRS, R, T, E, F= cv2.stereoCalibrate(objpoints,
                                                            imgpointsL,
                                                            imgpointsR,
                                                            mtxL,
                                                            distL,
                                                            mtxR,
                                                            distR,
                                                            chess_img_right.shape[::-1],
                                                            criteria = criteria_stereo,
                                                            flags = cv2.CALIB_FIX_INTRINSIC)

    # StereoRectify function
    rectify_scale = 0 # if 0 image croped, if 1 image nor croped
    RL, RR, PL, PR, Q, roiL, roiR= cv2.stereoRectify(MLS, dLS, MRS, dRS,
                                                    chess_img_right.shape[::-1], R, T,
                                                    rectify_scale,(0,0))  # last paramater is alpha, if 0= croped, if 1= not croped
    # initUndistortRectifyMap function
    left_stereo_map = cv2.initUndistortRectifyMap(MLS, dLS, RL, PL,
                                                  chess_img_right.shape[::-1], 
                                                  cv2.CV_16SC2)   # cv2.CV_16SC2 this format enables us the programme to work faster
    right_stereo_map = cv2.initUndistortRectifyMap(MRS, dRS, RR, PR,
                                                   chess_img_right.shape[::-1], 
                                                   cv2.CV_16SC2)
    
    return left_stereo_map, right_stereo_map


def stereo_builder(window_size = 3, min_disp = 2):
    num_disp = 130-min_disp
    stereo = cv2.StereoSGBM_create(minDisparity = min_disp,
        numDisparities = num_disp,
        blockSize = window_size,
        uniquenessRatio = 10,
        speckleWindowSize = 100,
        speckleRange = 32,
        disp12MaxDiff = 5,
        P1 = 8*3*window_size**2,
        P2 = 32*3*window_size**2)
    
    stereo_left = stereo
    stereo_right= cv2.ximgproc.createRightMatcher(stereo)
    
    return stereo, stereo_left, stereo_right 


def wls_filter_builder(stereo, lmbda=80000, sigma=1.8, visual_multiplier=1.0):
    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=stereo)
    wls_filter.setLambda(lmbda)
    wls_filter.setSigmaColor(sigma)
    return wls_filter



def preprocess_frame(frame_left, frame_right, left_stereo_map, right_stereo_map):
    left_nice = cv2.remap(frame_left, left_stereo_map[0], left_stereo_map[1], 
                          interpolation = cv2.INTER_LANCZOS4, 
                          borderMode = cv2.BORDER_CONSTANT)  # Rectify the image using the kalibration parameters founds during the initialisation
    right_nice = cv2.remap(frame_right, right_stereo_map[0], right_stereo_map[1], 
                           interpolation = cv2.INTER_LANCZOS4, 
                           borderMode = cv2.BORDER_CONSTANT)
    
    # Convert from color(BGR) to gray
    gray_right = cv2.cvtColor(right_nice,cv2.COLOR_BGR2GRAY)
    gray_left = cv2.cvtColor(left_nice,cv2.COLOR_BGR2GRAY)

    return gray_left, gray_right

def disp_builder( gray_left, gray_right, stereo_left, stereo_right):
     # Compute the 2 images for the Depth_image
    disp  = stereo_left.compute(gray_left, gray_right)#.astype(np.float32)/ 16
    disp_left = disp
    disp_right = stereo_right.compute(gray_right, gray_left)
    disp_left = np.int16(disp_left)
    disp_right = np.int16(disp_right)
    
    return disp, disp_left, disp_right

def apply_wls_filter(wls_filter, disp, disp_left, disp_right, gray_left, gray_right):
    # Using the WLS filter
    filtered_img = wls_filter.filter(disp_left, gray_left, None, disp_right)
    filtered_img = cv2.normalize(src=filtered_img, dst=filtered_img, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)
    filtered_img = np.uint8(filtered_img)
    #cv2.imshow('Disparity Map', filteredImg)
    disp = ((disp.astype(np.float32)/ 16)-min_disp)/num_disp # Calculation allowing us to have 0 for the most distant object able to detect

##    # Resize the image for faster executions
##    dispR= cv2.resize(disp,None,fx=0.7, fy=0.7, interpolation = cv2.INTER_AREA)

    # Filtering the Results with a closing filter
    closing= cv2.morphologyEx(disp,cv2.MORPH_CLOSE, kernel) # Apply an morphological filter for closing little "black" holes in the picture(Remove noise) 

    # Colors map
    dispc = (closing-closing.min())*255
    dispc = dispc.astype(np.uint8)                                   # Convert the type of the matrix from float32 to uint8, this way you can show the results with the function cv2.imshow()
    disp_color= cv2.applyColorMap(dispc, cv2.COLORMAP_OCEAN)         # Change the Color of the Picture into an Ocean Color_Map
    filt_color= cv2.applyColorMap(filtered_img, cv2.COLORMAP_OCEAN) 
    
    return filt_color, disp_color


def preprocess_video_frame(frame_left, frame_right, left_stereo_map, right_stereo_map, stereo_left, stereo_right, wls_filter):
    gray_left, gray_right = preprocess_frame(frame_left, frame_right, left_stereo_map, right_stereo_map)
    disp, disp_left, disp_right = disp_builder(gray_left, gray_right, stereo_left, stereo_right)
    filt_color, disp_color = apply_wls_filter(wls_filter, disp, disp_left, disp_right, gray_left, gray_right)
    
    return filt_color
    
    


