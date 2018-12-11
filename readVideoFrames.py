'''
import cv2
vidcap = cv2.VideoCapture('5C 2017 SPA (JEREZ) bandera.mp4')
success,image = vidcap.read()
count = 0
success = True
while success:
  cv2.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file
  success,image = vidcap.read()
  print('Read a new frame: ', success)
  count += 1
'''

import cv2
import numpy as np
import math
from scipy import ndimage

vidcap = cv2.VideoCapture('vid')
success,image = vidcap.read()
count = 0
success = True

while success:
  success,image = vidcap.read()
  vidcap.set(1, count)
  #cv2.imwrite("frame%d.jpeg" % count, image)     # save frame as JPEG file
  #img_before = cv2.imread("frame%d.jpeg" % count)
  img_before = image.astype(np.uint8)

  print('Read a new frame: ', success)

  img_gray = cv2.cvtColor(img_before, cv2.COLOR_BGR2GRAY)
  img_edges = cv2.Canny(img_gray, 100, 100, apertureSize=3)
  lines = cv2.HoughLinesP(img_edges, 1, math.pi / 180.0, 100, minLineLength=100, maxLineGap=5)

  #angles = []

  for x1, y1, x2, y2 in lines[0]:
      cv2.line(img_before, (x1, y1), (x2, y2), (255, 0, 0), 3)
      angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
      #angles.append(angle)

  #median_angle = np.median(angle)
  #img_rotated = ndimage.rotate(img_before, angle)

  print "Angle is {}".format(angle)
  #cv2.imwrite('rotated.jpg', img_rotated)

  count += 10
