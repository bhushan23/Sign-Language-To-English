import cv2
import numpy as np
from interface import *

cam = cv2.VideoCapture(0)

cv2.namedWindow("test")

img_counter = 0

backSub = cv2.createBackgroundSubtractorMOG2()

detector = SignLangDetector()
while True:
    ret, frame = cam.read()
    # cv2.imshow("test", frame)

    img = frame #np.random.randint(0, 256, size=(200, 300, 3), dtype=np.uint8)
    height, width, channels = img.shape
    upper_left = (int(width / 4), int(height * .1))
    bottom_right = (int(width * 3 / 4), int(height * 0.9))

    # draw in the image
    cv2.rectangle(img, upper_left, bottom_right, (0, 255, 0), 2)
    # fgMask = backSub.apply(img)
    # cv2.imshow('test', img)
    # img2 = np.moveaxis(img, -1, 0)   
    # img2 = fgMask * img2
    # img2 = np.moveaxis(img2, 0, -1) 
    # cv2.imshow('mask', img2)
    # cv2.waitKey(), 

    # indexing array 
    rect_img = img[upper_left[1] : bottom_right[1], upper_left[0] : bottom_right[0]]
    # fgMask = backSub.apply(rect_img)
    

    # img2 = np.moveaxis(rect_img, -1, 0)   
    # img2 = fgMask * img2
    # img2 = np.moveaxis(img2, 0, -1) 
    # cv2.imshow('test', img2)
    # cv2.waitKey(), 

    img_name = "temp_opencv/opencv_frame_{}.png".format(img_counter)
    cv2.imwrite(img_name, frame)

    out = detector.predict(img_name)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, out,(10,300), font, 4,(255,255,255), 2, cv2.LINE_AA)
    print('Prediction: ', out)
    cv2.imshow('img', img)

    if not ret:
        break
    k = cv2.waitKey(2)

    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        img_name = "temp_opencv/opencv_frame_{}.png".format(img_counter)
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        img_counter += 1

cam.release()

cv2.destroyAllWindows()