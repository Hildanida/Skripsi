import tensorflow as tf
from utils import backbone
from api import object_counting_api_dua
import numpy as np
import cv2
import os

import pygame
from pygame import mixer



input_video = 2
cap = cv2.VideoCapture(2)

input_video_dua = 1
cap2 = cv2.VideoCapture(1)

detection_graph, category_index = backbone.set_model('ssd_mobilenet_v1_coco_2018_01_28')#ssd_mobilenet_v1_coco_2017_11_17 #ssd_mobilenet_v1_coco_2018_01_28

while True:
  ret, frame = cap.read()
  gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
  
  ret2, frame2 = cap2.read()
  gray2 = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
  
  targeted_objects = "mobil, sepeda motor, sepeda, meja, kursi, manusia, mejo, MEJA" # (for counting targeted objects) change it with your targeted objects
  fps = 25 # change it with your input video fps
  width = 300 # change it with your input video width
  height = 300 # change it with your input video height
  is_color_recognition_enabled = 0

  object_counting_api_dua.targeted_object_counting(input_video, detection_graph, category_index, is_color_recognition_enabled, targeted_objects, fps, width, height) # targeted objects counting
  
  object_counting_api_dua.targeted_object_counting(input_video_dua, detection_graph, category_index, is_color_recognition_enabled, targeted_objects, fps, width, height) # targeted objects counting
  
  
  
  if cv2.waitKey(1) & 0xFF ==ord('q'):
      cap.release()
      cap2.release()
      frame.release()
      frame2.release()
      cv2.destroyAllWindows()
      break
      
      