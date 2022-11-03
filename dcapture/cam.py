import cv2 as cv
import numpy as np
import fire
import random
import os
from pathlib import Path



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

def generate_footer(frame_top, text_info="Status Information"):
    flh, flw, fld = frame_top.shape
    footer = bar_info(img_size=(flh, flw), text=text_info)
    dual = np.concatenate((frame_top, footer), axis=0)

    return dual


def save_to_dir(dst, frame_left, frame_right):
    rstr = str(random.random())[2:8]
    base_path = Path(dst)
    path = base_path.joinpath(f'take_{rstr}')
    path.mkdir(exist_ok = True, parents=True)

    left_path = path.joinpath('left_cam.jpg')
    right_path = path.joinpath('right_cam.jpg')

    cv.imwrite(str(left_path), frame_left)
    cv.imwrite(str(right_path), frame_right)

    return path, left_path, right_path

    

def stereo_vision_run(cam_left, cam_right, target_dir):
    """_summary_

    Args:
        cam_left (_type_): _description_
        cam_right (_type_): _description_
        target_dir (_type_): _description_
    """
    info_bar_text= f'Directory Saved Status: {target_dir}'
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

        window = generate_footer(window, text_info=info_bar_text)
        
        if (ret_left == True) and (ret_right == True):
            cv.imshow('Frame', window)
            if cv.waitKey(25) & 0xFF == ord('q'):
                break

            if cv.waitKey(25) & 0xFF == ord('t'):
                paths = save_to_dir(target_dir, frame_left, frame_right)
                info_bar_text = f"Last Saved : {paths[0]}"

                pass
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