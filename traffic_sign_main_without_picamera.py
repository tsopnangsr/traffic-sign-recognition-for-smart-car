#!/usr/bin/env python
# coding: utf-8
# Date: 2019/11/18
# Time: 03:52
# Traffic Sign detection and classification
    # by Modeniac Lucas, FGI Douala Students.

# import the necessary packages
from __future__ import division
from matplotlib import pyplot as plt
from math import cos, sin
from PIL import Image
from time import sleep
from tensorflow.keras.models import load_model
from skimage import transform
from skimage import exposure
from skimage import io
from imutils import paths
import numpy as np
import argparse
import imutils
import random
import cv2
import os


# load the traffic sign recognizer model
print("[INFO] loading model...")
model = load_model("output/trafficsignnet.model")


# load the label names
labelNames = open("signnames.csv").read().strip().split("\n")[1:]
labelNames = [l.split(",")[1] for l in labelNames]

#import serial

#ser = serial.Serial('/dev/ttyACM0', 9600) # Establish the connection on a specific port
#ser = serial.Serial('COM13', 9600) # Establish the connection on a specific port
counter = 32 # Below 32 everything in ASCII is gibberish



cap = cv2.VideoCapture(0) # moi crÃ©e obj video


green = (0, 255, 0)

def show(image):
    # Figure size in inches
    plt.figure(figsize=(10, 10))

    # Show image, with nearest neighbour interpolation
    plt.imshow(image, interpolation='nearest')

def overlay_mask(mask, image):
	#make the mask rgb
    rgb_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    #calculates the weightes sum of two arrays. in our case image arrays
    #input, how much to weight each.
    #optional depth value set to 0 no need
    img = cv2.addWeighted(rgb_mask, 0.5, image, 0.5, 0)
    return img

def find_biggest_contour(image):
    # Copy
    image = image.copy()
    #input, gives all the contours, contour approximation compresses horizontal,
    #vertical, and diagonal segments and leaves only their end points. For example,
    #an up-right rectangular contour is encoded with 4 points.
    #Optional output vector, containing information about the image topology.
    #It has as many elements as the number of contours.
    #we dont need it
    contours, hierarchy = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    #print(contours)

    # Isolate largest contour
    contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
    biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]

    mask = np.zeros(image.shape, np.uint8)
    cv2.drawContours(mask, [biggest_contour], -1, 255, -1)
    return biggest_contour, mask

def circle_contour(image, contour):
    # Bounding ellipse
    image_with_ellipse = image.copy()
    #easy function
    ellipse = cv2.fitEllipse(contour)
    #add it LINE_AA
    cv2.ellipse(image_with_ellipse, ellipse, green, 2, cv2.LINE_AA)
    #cv2.ellipse(image_with_ellipse, ellipse, green, 2, cv2.CV_AA)
    #print(ellipse)
    return image_with_ellipse

def find_strawberry(image):
    #RGB stands for Red Green Blue. Most often, an RGB color is stored
    #in a structure or unsigned integer with Blue occupying the least
    #significant “area” (a byte in 32-bit and 24-bit formats), Green the
    #second least, and Red the third least. BGR is the same, except the
    #order of areas is reversed. Red occupies the least significant area,
    # Green the second (still), and Blue the third.
    # we'll be manipulating pixels directly
    #most compatible for the transofrmations we're about to do
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Make a consistent size
    #get largest dimension
    max_dimension = max(image.shape)
    #The maximum window size is 700 by 660 pixels. make it fit in that
    scale = 700/max_dimension
    #resize it. same width and hieght none since output is 'image'.
    image = cv2.resize(image, None, fx=scale, fy=scale)

    #we want to eliminate noise from our image. clean. smooth colors without
    #dots
    # Blurs an image using a Gaussian filter. input, kernel size, how much to filter, empty)
    image_blur = cv2.GaussianBlur(image, (7, 7), 0)
    #t unlike RGB, HSV separates luma, or the image intensity, from
    # chroma or the color information.
    #just want to focus on color, segmentation
    image_blur_hsv = cv2.cvtColor(image_blur, cv2.COLOR_RGB2HSV)

    # Filter by colour
    # 0-10 hue
    #minimum red amount, max red amount
    min_red = np.array([0, 100, 80])
    max_red = np.array([10, 256, 256])
    #layer
    mask1 = cv2.inRange(image_blur_hsv, min_red, max_red)

    #birghtness of a color is hue
    # 170-180 hue
    min_red2 = np.array([170, 100, 80])
    max_red2 = np.array([180, 256, 256])
    mask2 = cv2.inRange(image_blur_hsv, min_red2, max_red2)

    #looking for what is in both ranges
    # Combine masks
    mask = mask1 + mask2

    # Clean up
    #we want to circle our strawberry so we'll circle it with an ellipse
    #with a shape of 15x15
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    #morph the image. closing operation Dilation followed by Erosion.
    #It is useful in closing small holes inside the foreground objects,
    #or small black points on the object.
    mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    #erosion followed by dilation. It is useful in removing noise
    mask_clean = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel)

    # Find biggest strawberry
    #get back list of segmented strawberries and an outline for the biggest one
    big_strawberry_contour, mask_strawberries = find_biggest_contour(mask_clean)

    # Overlay cleaned mask on image
    # overlay mask on image, strawberry now segmented
    overlay = overlay_mask(mask_clean, image)

    # Circle biggest strawberry
    #circle the biggest one
    circled = circle_contour(overlay, big_strawberry_contour)
    show(circled)

##
##
##    print(big_strawberry_contour[:,:,0])
##    print(mask_strawberries)
##
##    max_x_value = np.max(big_strawberry_contour[:,:,0])
##    min_x_value = np.min(big_strawberry_contour[:,:,0])
##    max_y_value = np.max(big_strawberry_contour[:,:,1])
##    min_y_value = np.min(big_strawberry_contour[:,:,1])
##    print(max_x_value)
##    print(min_x_value)
##    print(max_y_value)
##    print(min_y_value)
##    print(image.shape)
##    #cv2.rectangle(mask_strawberries , (10, 10), (100, 100), (155, 0, 0), 5)
##    cv2.rectangle(image , (min_x_value, min_y_value), (max_x_value, max_y_value), (155, 0, 0), 10)
##    cv2.line(image , (min_x_value, max_x_value), (min_y_value, max_y_value), (155, 0, 0), 10)
##    plt.imshow(image, cmap='gray', interpolation='bicubic')
##    plt.show()

    #big_strawberry_contour.reshape([157,2])
##    print("La matrice complete")
##    print(big_strawberry_contour)
##    print("colonne 1")
##    print(big_strawberry_contour[1:])(2)
##    print("colonne 2")
##    print(big_strawberry_contour[:1])
##    print("Taille de la matrice")
    #big_strawberry_contour.reshape(278, 2)
    #rect = cv2.minAreaRect(big_strawberry_contour)

    ##    plt.plot([min_x_value, max_x_value], [min_y_value, max_y_value], 'c', linewidth=5)
##    cropped_img = circled[min_x_value:max_x_value, min_y_value:max_y_value]
##    print(cropped_img)
##    plt.plot(cropped_img, cmap='gray', interpolation='bicubic')
##    plt.show()


    #we're done, convert back to original color scheme
    bgr = cv2.cvtColor(circled, cv2.COLOR_RGB2BGR)

    return bgr, mask_strawberries, big_strawberry_contour


def detect(cap):

    font = cv2.FONT_HERSHEY_SIMPLEX
    ret, img = cap.read() # moi acquerir image
    #img = cv2.imread(filepath+file)

    #Computer Vision part, Detection of ROI
    image1 = img
    result, mask_panel, big_panel_contour = find_strawberry(image1)
    img_panel = cv2.bitwise_and(result, result, mask=mask_panel)


    ###########################################
    #Cropping image :: Rognage de l'image
    ###########################################
    max_x_value = np.max(big_panel_contour[:,:,0])
    min_x_value = np.min(big_panel_contour[:,:,0])
    max_y_value = np.max(big_panel_contour[:,:,1])
    min_y_value = np.min(big_panel_contour[:,:,1])

    cropped_img = result[min_y_value:max_y_value, min_x_value:max_x_value]

    # Processing the ROI using CNN DeepLearning
    image = cropped_img
    image = transform.resize(image, (32, 32))
    image = exposure.equalize_adapthist(image, clip_limit=0.1)
    # preprocess the image by scaling it to the range [0, 1]
    image = image.astype("float32") / 255.0
    image = np.expand_dims(image, axis=0)
    # make predictions using the traffic sign recognizer CNN
    preds = model.predict(image)
    j = preds.argmax(axis=1)[0]
    label = labelNames[j]

    # load the image using OpenCV, resize it, and draw the label
	# on it
    image = img
    image = imutils.resize(image, width=128)
    cv2.putText(image, label, (5, 15), cv2.FONT_HERSHEY_SIMPLEX,
		0.45, (0, 0, 255), 2)
    #cv2.putText(cimg,'YELLOW',(i[0], i[1]), font, 1,(255,0,0),2,cv2.LINE_AA)
    cv2.imshow('detected results', image)
    ##cv2.imwrite(path+'//result//'+file, cimg)
    # cv2.imshow('maskr', maskr)
    # cv2.imshow('maskg', maskg)
    # cv2.imshow('masky', masky)



if __name__ == '__main__':

    while(True):
        detect(cap)
        counter +=1
        #ser.write(str(chr(counter))) # Convert the decimal number to ASCII then send it to the Arduino
        #print(str(ser.readline())) # Read the newest output from the Arduino
        #print(float(ser.readline().split('*')[0]))
        #sleep(.1) # Delay for one tenth of a second
        if counter == 255:
            counter = 32
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
