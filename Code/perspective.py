import cv2
import numpy as np

img_1 = cv2.imread('../Data/Frames_project_video/frame40.jpg')
# img = cv2.imread('ROI.jpg')

img = img_1[400 : img_1.shape[0], 0 : img_1.shape[1]]

pts_src = np.load('points.npy')
pts_src = np.float32(pts_src)
# print(pts_src)
pts_dst = np.float32([(0, 0), (199, 0), (199, 199), (0, 199)])

h = cv2.getPerspectiveTransform(pts_src, pts_dst)
    
im_out = cv2.warpPerspective(img, h, (200, 200))

# im_gray = cv2.cvtColor(im_out, cv2.COLOR_BGR2GRAY)

# _, thresh = cv2.threshold(im_gray, 127, 255, cv2.THRESH_BINARY)

# sobelx64f = cv2.Sobel(im_gray, cv2.CV_64F, 1, 0, ksize=5)
# abs_sobelx64f = np.absolute(sobelx64f)
# sobelx_8u = np.uint8(abs_sobelx64f)

# sobely64f = cv2.Sobel(im_gray, cv2.CV_64F, 0, 1, ksize=5)
# abs_sobely64f = np.absolute(sobely64f)
# sobely_8u = np.uint8(abs_sobely64f)
""" 
Morphological operation to get the skeleton structure 
"""

# kernel = np.ones((4, 4),np.uint8)
# dilation = cv2.dilate(im_gray, kernel, iterations = 1)
# closing = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel)
# erosion = cv2.erode(closing, kernel, iterations = 3)

# images = np.hstack((im_gray, dilation, closing, erosion))

""" 
probabilistic Hough lines 
"""

# lines = cv2.HoughLinesP(erosion, 1, np.pi/180, 180, minLineLength=100, maxLineGap=250)

# for line in lines:
#     x1, y1, x2, y2 = line[0]
#     cv2.line(im_out, (x1, y1), (x2, y2), (0, 0, 255), 1)


cv2.imshow('ROI',  img)
cv2.imshow('LinesY',  img_1)
# cv2.imwrite('lane_ero.jpg', erosion)
cv2.imshow('warpPerspective',  im_out)
# cv2.imwrite('ROI.jpg', ROI)
while (1):
	k = cv2.waitKey(1) & 0xFF
	if k == 27:
		break

cv2.destroyAllWindows()