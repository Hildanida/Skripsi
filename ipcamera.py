import numpy as np
import cv2
import requests

cap = "http://192.168.137.11:8080/shot.jpg"

while True:
  
  img_resp = requests.get(cap)
  img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
  img = cv2.imdecode(img_arr, -1)
  
  cv2.imshow("Kameraaa", img)

  if cv2.waitKey(1) & 0xFF ==ord('q'):
      cap.release()
      cv2.destroyAllWindows()
      break
