import cv2
import numpy as np
from PIL import Image   


# Reading the image
img = cv2.imread('C:\Users\manas\Downloads\debri.jpg')

# convert to hsv colorspace
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# lower bound and upper bound for Green color
lower_bound = np.array([50, 20, 20])	 
upper_bound = np.array([100, 255, 255])

# find the colors within the boundaries
mask = cv2.inRange(hsv, lower_bound, upper_bound)

#define kernel size  
kernel = np.ones((7,7),np.uint8)

# Remove unnecessary noise from mask

mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

# Segment only the detected region
segmented_img = cv2.bitwise_and(img, img, mask=mask)

# Find contours from the mask

contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

output = cv2.drawContours(segmented_img, contours, -1, (0, 0, 255), 3)

# Showing the output

cv2.imshow("Output", output)
# lower bound and upper bound for Yellow color

lower_bound = np.array([20, 80, 80])	 
upper_bound = np.array([30, 255, 255])

# Draw contour on original image

output = cv2.drawContours(img, contours, -1, (0, 0, 255), 3)

cv2.imshow("Output", output)

cv2.waitKey(0)
cv2.destroyAllWindows()