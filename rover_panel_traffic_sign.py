# -*- coding: utf-8 -*-
from __future__ import division
import cv2
#to show the image
from matplotlib import pyplot as plt
import numpy as np
from math import cos, sin

from PIL import Image
import pytesseract

#import picamera
import time
import os

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

#read the image
###########################################
#Picture capture by the camera
###########################################
##camera = picamera.PiCamera()
##picture_name = ("test-image-picamera"+str(1)+".jpg")
###picture_name = ("/home/pi/TSR_Python/Radar_de_feu/infractions/test-image-picamera"+str(1)+".jpg")
###camera . capture("test-image-picamera.jpg")
##camera . capture(picture_name)
##panel_num = 1
##cmd_fswebcam = "fswebcam -r 1280x720 --no-banner /home/pi/Documents/TSR_Folder/TSR_Python_New/Vision/rover_img/captured_panel/"
##picture_name = ("panel"+str(panel_num)+"_1.jpg")
###camera.capture(picture_name)
##os.system(cmd_fswebcam + picture_name)
##os.system("gpicview " +"/home/pi/Documents/TSR_Folder/TSR_Python_New/Vision/rover_img/captured_panel/"+ picture_name)
##
#############################################
###Color-Shape detection :: couleur et forme
###########################################
#image = cv2.imread('images/berry.jpg')
image = cv2.imread('rover_img/route_70.jpeg', cv2.IMREAD_COLOR)
#image = cv2.imread(picture_name)
#detect it
result, mask_panel, big_panel_contour = find_strawberry(image)
img_panel = cv2.bitwise_and(result, result, mask=mask_panel)
#write the new image
#cv2.imwrite('images/berry2.jpg', result)
cv2.imwrite('rover_img/route_70_2.jpeg', result)

##plt.imshow(result, cmap='gray', interpolation='bicubic')
##plt.show()

###########################################
#Cropping image :: Rognage de l'image
###########################################
print(big_panel_contour[:,:,0])
print(mask_panel)

max_x_value = np.max(big_panel_contour[:,:,0])
min_x_value = np.min(big_panel_contour[:,:,0])
max_y_value = np.max(big_panel_contour[:,:,1])
min_y_value = np.min(big_panel_contour[:,:,1])
print(max_x_value)
print(min_x_value)
print(max_y_value)
print(min_y_value)
print(result.shape)
##cv2.rectangle(result , (10, 10), (100, 100), (155, 0, 0), 5)
##cv2.rectangle(result , (min_x_value, min_y_value), (max_x_value, max_y_value), (155, 0, 0), 10)
##cv2.line(result , (min_x_value, max_x_value), (min_y_value, max_y_value), (155, 0, 0), 10)
plt.imshow(result, cmap='gray', interpolation='bicubic')
plt.show()


cropped_img = result[min_y_value:max_y_value, min_x_value:max_x_value]

print("result")
print(result)
plt.imshow(cropped_img, cmap='gray', interpolation='bicubic')
##plt.imshow(result[min_x_value:max_x_value, min_y_value:max_y_value, :], cmap='gray', interpolation='bicubic')
plt.show()

###########################################
#Gray level scale :: Image a niveau de gris
###########################################
img = result
##img=cv2.imread("rover_img/route_90.jpeg",1)
##cv2.namedWindow("img",cv2.WINDOW_NORMAL)

gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#adaptiveThreshold(src, maxValue, adaptiveMethod, thresholdType, blockSize, C)
gaus=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,115,1)
##ret,threshold=cv2.threshold(img,12,255,cv2.THRESH_BINARY)
ret,threshold=cv2.threshold(gray,0, 100,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)

###########################################
#Edge detection :: Detection de contour
###########################################
boundary = threshold
laplacian = cv2.Laplacian(boundary, cv2.CV_64F)
sobelx = cv2.Sobel(boundary, cv2.CV_64F, 1, 0, ksize = 5)
sobely = cv2.Sobel(boundary, cv2.CV_64F, 0, 1, ksize = 5)
edges = cv2.Canny(boundary, 100, 200)

##print(gaus.shape)
##
cv2.imwrite('rover_img/route_'+str(70)+'_threshold.jpeg', threshold)
cv2.imwrite('rover_img/route_'+str(70)+'_gray.jpeg', gray)
cv2.imwrite('rover_img/route_'+str(70)+'_img.jpeg', img)
cv2.imwrite('rover_img/route_'+str(70)+'_gaus.jpeg', gaus)
cv2.imwrite('rover_img/route_'+str(70)+'_laplacian.jpeg', laplacian)
cv2.imwrite('rover_img/route_'+str(70)+'_sobelx.jpeg', sobelx)
cv2.imwrite('rover_img/route_'+str(70)+'_sobely.jpeg', sobely)
cv2.imwrite('rover_img/route_'+str(70)+'_edges.jpeg', edges)

plt.imshow(threshold, cmap='gray', interpolation='bicubic')
plt.show()
plt.imshow(gray, cmap='gray', interpolation='bicubic')
plt.show()
plt.imshow(img, cmap='gray', interpolation='bicubic')
plt.show()
plt.imshow(gaus, cmap='gray', interpolation='bicubic')
plt.show()

plt.imshow(laplacian, cmap='gray', interpolation='bicubic')
plt.show()
plt.imshow(sobelx, cmap='gray', interpolation='bicubic')
plt.show()
plt.imshow(sobely, cmap='gray', interpolation='bicubic')
plt.show()
plt.imshow(edges, cmap='gray', interpolation='bicubic')
plt.show()

##cv2.imshow('Original',frame)
##cv2.imshow('Mask',mask)
##cv2.imshow('laplacian',laplacian)
##cv2.imshow('sobelx',sobelx)
##cv2.imshow('sobely',sobely)


#im = Image.open("rover_img/panneau_30.jpg")
im = img_panel
##plt.imshow(img, cmap='gray', interpolation='bicubic')
##plt.show()
text = pytesseract.image_to_string(im, lang = 'eng')
print('image: '+ text)
text = pytesseract.image_to_string(threshold, lang = 'eng')
print('threshold: '+ text)
text = pytesseract.image_to_string(gray, lang = 'eng')
print('gray: '+ text)
text = pytesseract.image_to_string(img, lang = 'eng')
print('img: '+ text)
text = pytesseract.image_to_string(gaus, lang = 'eng')
print('gaus: '+ text)
text = pytesseract.image_to_string(laplacian, lang = 'eng')
print('laplacian: '+ text)
text = pytesseract.image_to_string(sobelx, lang = 'eng')
print('sobelx: '+ text)
text = pytesseract.image_to_string(sobely, lang = 'eng')
print('sobely: '+ text)
text = pytesseract.image_to_string(edges, lang = 'eng')
print('edges: '+ text)
##
cv2.imwrite('rover_img/route_'+str(70)+'_threshold.jpeg', threshold)
cv2.imwrite('rover_img/route_'+str(70)+'_gray.jpeg', gray)
cv2.imwrite('rover_img/route_'+str(70)+'_img.jpeg', img)
cv2.imwrite('rover_img/route_'+str(70)+'_gaus.jpeg', gaus)
cv2.imwrite('rover_img/route_'+str(70)+'_laplacian.jpeg', laplacian)
cv2.imwrite('rover_img/route_'+str(70)+'_sobelx.jpeg', sobelx)
cv2.imwrite('rover_img/route_'+str(70)+'_sobely.jpeg', sobely)
cv2.imwrite('rover_img/route_'+str(70)+'_edges.jpeg', edges)
