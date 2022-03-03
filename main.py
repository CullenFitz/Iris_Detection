import numpy as np
import cv2 as cv

import cv2
import numpy as np

img1 = [r"C:\Users\Cullen\Downloads\iris_matlab\iris1.jpg", 70, 100]
img2 = [r"C:\Users\Cullen\Downloads\iris_matlab\iris2.jpg", 50, 70]
img3 = [r"C:\Users\Cullen\Downloads\iris_matlab\iris3.jpg", 60, 100]
img4 = [r"C:\Users\Cullen\Downloads\iris_matlab\iris4.jpg", 60, 50]
img5 = [r"C:\Users\Cullen\Downloads\iris_matlab\iris5.jpg", 90, 110]
img6 = [r"C:\Users\Cullen\Downloads\iris_matlab\iris6.jpg", 80, 80]

imgArr = [img1, img2, img3, img4, img5, img6]

for i in imgArr:

    img = cv2.imread(i[0],0)
    img = cv2.medianBlur(img,5)
    img2 = cv2.imread(i[0])
    cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)

    circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,20,
                                param1=i[1],param2=i[2],minRadius=0,maxRadius=0)

    circles = np.uint16(np.around(circles))
    for j in circles[0,:]:
        # draw the outer circle
        cv2.circle(cimg,(j[0],j[1]),j[2],(0,255,0),2)
        # draw the center of the circle
        cv2.circle(cimg,(j[0],j[1]),2,(0,0,255),3)

        # crop eye from image
        print(j[0],j[1],j[2])
        mask1 = np.zeros_like(img2)
        mask1 = cv2.circle(mask1, (j[0], j[1]), j[2], (255, 255, 255), -1)
        mask2 = np.zeros_like(img)
        mask2 = cv2.circle(mask2, (j[0], j[1]), j[2], (255, 255, 255), -1)

        mask = cv2.subtract(mask2, mask1)

        result = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        result[:, :, 3] = mask[:, :, 0]

        cv2.imshow('detected circles', result)

    cv2.waitKey(0)
    cv2.destroyAllWindows()