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

    height, width = img.shape
    mask = np.zeros((height, width), np.uint8)

    counter = 0

    circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,20,
                                param1=i[1],param2=i[2],minRadius=0,maxRadius=0)

    circles = np.uint16(np.around(circles))
    for j in circles[0,:]:

        # draw the outer circle
        cv2.circle(cimg,(j[0],j[1]),j[2],(0,255,0),2)
        # draw the center of the circle
        cv2.circle(cimg,(j[0],j[1]),2,(0,0,255),3)

        cv2.circle(mask, (j[0], j[1]), j[2], (255, 255, 255), -1)
        masked_data = cv2.bitwise_and(cimg, cimg, mask=mask)

        # Apply Threshold
        _, thresh = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)

        cnt = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

        # print len(contours)
        x, y, w, h = cv2.boundingRect(cnt[0])

        # Crop masked_data
        crop = masked_data[y:y + h, x:x + w]

        cv2.imshow('detected circles', crop)

    cv2.waitKey(0)
    cv2.destroyAllWindows()