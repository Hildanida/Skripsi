import cv2 as cv
import numpy as np
import fire




def bar_info(img_size, text, height=100,
             font=cv.FONT_HERSHEY_SIMPLEX, 
             font_scale=1, thickness=2,
             bg_color=(0,0,0), color=(255,255,255)):
    
    h,w = img_size
    image = np.zeros((height, w, 3), np.uint8)
    image[:0:w] = bg_color
    # get boundary of this text
    textsize = cv.getTextSize(text, font, font_scale, thickness)[0]
    # get coords based on boundary
    textX = int(image.shape[1] - textsize[0]) // 2
    textY = int(image.shape[0] + textsize[1]) // 2
    
    # print(textX,textY)

    image = cv.putText(image, text, (textX, textY), font, font_scale, color, thickness, cv.LINE_AA)
    return image


def generate_window(frame_left, frame_right):
    
    flh, flw, fld = frame_left.shape
    top_left_image = bar_info(img_size=(flh, flw), text="Left Camera")
    left = np.concatenate((top_left_image, frame_left), axis=0)
    
    frh, frw, frd = frame_right.shape
    top_right_image = bar_info(img_size=(frh, frw), text="Right Camera")
    right = np.concatenate((top_right_image, frame_right), axis=0)
    
    
    dual = np.concatenate((left, right), axis=1)
    h,w,d = dual.shape
    
    
    return dual
    
    
    
    
    

def stereo_vision_run(cam_left, cam_right, target_dir):
    """_summary_

    Args:
        cam_left (_type_): _description_
        cam_right (_type_): _description_
        target_dir (_type_): _description_
    """
    # Create a VideoCapture object
    cap_left = cv.VideoCapture(cam_left)
    cap_right = cv.VideoCapture(cam_right)
    # Check if camera opened successfully
    if (cap_left.isOpened() == False) or (cap_right.isOpened() == False):
        print("Error opening video stream or file")
    # Read until video is completed
    while (cap_left.isOpened() and cap_right.isOpened()):
        # Capture frame-by-frame
        ret_left, frame_left = cap_left.read()
        ret_right, frame_right = cap_right.read()
        
        
        # window = np.concatenate((frame_left, frame_right), axis=1)
        window = generate_window(frame_left, frame_right)
        
        
        if (ret_left == True) and (ret_right == True):
            cv.imshow('Frame', window)
            if cv.waitKey(25) & 0xFF == ord('q'):
                break
        # Break the loop
        else:
            break
    # When everything done, release the video capture object
    cap_left.release()
    cap_right.release()
    # Closes all the frames
    cv.destroyAllWindows()







if __name__ == '__main__':
  fire.Fire(stereo_vision_run)