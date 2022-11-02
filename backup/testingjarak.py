import tensorflow as tf
from utils import backbone
from api import object_counting_api_jarak
import numpy as np
import cv2
import os
from playsound import playsound

from utils import detector_utils as detector_utils

input_video = 0
cap = cv2.VideoCapture(0)
# By default I use an "SSD with Mobilenet" model here. See the detection model zoo (https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.
detection_graph, category_index = backbone.set_model('ssd_mobilenet_v1_coco_2018_01_28')#ssd_mobilenet_v1_coco_2017_11_17

while True:
  ret, frame = cap.read()
  gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
  
  targeted_objects = "mobil, sepeda motor, sepeda, meja, kursi, manusia" # (for counting targeted objects) change it with your targeted objects
  fps = 25 # change it with your input video fps
  width = 300 # change it with your input video width
  height = 300 # change it with your input video height
  is_color_recognition_enabled = 0

  object_counting_api_jarak.targeted_object_counting(input_video, detection_graph, category_index, is_color_recognition_enabled, targeted_objects, fps, width, height) # targeted objects counting
  
  object_counting_api_jarak.object_classes(input_video, detection_graph, category_index, is_color_recognition_enabled, targeted_objects, fps, width, height)
  
  
  
  if cv2.waitKey(1) & 0xFF ==ord('q'):
      cap.release()
      cv2.destroyAllWindows()
      break