import cv2
import os

videowriter = cv2.VideoWriter("test.avi", cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 20, (640, 256))
for i in range(4701):
    file = 'images/{}.jpg'.format(i)
    img = cv2.imread(file)
    videowriter.write(img)