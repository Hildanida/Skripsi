import sys
import numpy as np
import cv2

from kivy.app import App
from kivy.uix.widget import Widget
from kivy.graphics.texture import Texture
from kivy.graphics import Rectangle

from kivy.clock import Clock
from kivy.core.window import Window

import datetime

WINDOW_WIDTH = 1500
WINDOW_HEIGHT = 1008
### Full HD
# WINDOW_WIDTH = 1920
# WINDOW_HEIGHT = 1080
### 4K, UHD, UHDTV
# WINDOW_WIDTH = 3840
# WINDOW_HEIGHT = 2160

### Setting of the window
Window.size = (WINDOW_WIDTH, WINDOW_HEIGHT)

class MyApp(App):
        title = "opencv on kivy"

        def __init__(self, **kwargs):
            super(MyApp, self).__init__(**kwargs)
            Clock.schedule_interval(self.update, 1.0 / 30)
            self.widget = Widget()
            # self.cap = cv2.VideoCapture(0)
            self.cap = cv2.VideoCapture(6)
            # self.check_camera_connection()

        # The method by the intarval
        def update(self, dt):
            ret, org_img = self.cap.read()

            if org_img is None:
                print('load image')
                sys.exit(1)

            size = (900, 700)
            converted_img = cv2.resize(org_img, size)
            ### convert to gray
            # converted_img = cv2.cvtColor(converted_img, cv2.COLOR_BGR2GRAY)
            ### The color sequence in openCV is BGR, so fix it to RGB.
            converted_img = cv2.cvtColor(converted_img, cv2.COLOR_BGR2RGB)
            ### The origin of Kivy's coordinates is the lower left, so flip it upside down.
            converted_img = cv2.flip(converted_img, 0)
            ### OpenCV coordinate shape is "height", "width", and "channel" in that order Kivy's size is (width, height), so it needs to be reversed
            texture = Texture.create(size=(converted_img.shape[1], converted_img.shape[0]))
            texture.blit_buffer(converted_img.tostring())

            with self.widget.canvas:
               Rectangle(texture=texture ,pos=(0 + int(WINDOW_WIDTH/2) - int(converted_img.shape[1]/2), WINDOW_HEIGHT - converted_img.shape[0]), size=(converted_img.shape[1], converted_img.shape[0]))

            return self.widget

        def build(self):
            return self.widget

        def check_camera_connection(self):
            """
            Check the connection between any camera and the PC.
            """
            print('[', datetime.datetime.now(), ']', 'searching any camera...')
            true_camera_is = []

            for camera_number in range(0, 10):
                cap = cv2.VideoCapture(camera_number)
                ret, frame = cap.read()

                if ret is True:
                    true_camera_is.append(camera_number)
                    print("camera_number", camera_number, "Find!")

                else:
                    print("camera_number", camera_number, "None")
            print("The number of cameras connected to : ", len(true_camera_is))

if __name__ == '__main__':
    MyApp().run()