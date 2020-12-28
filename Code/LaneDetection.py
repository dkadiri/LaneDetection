import cv2
import numpy as np
import matplotlib.pyplot as plt


k  = np.array([[  1.15422732e+03, 0.00000000e+00, 6.71627794e+02],
 [  0.00000000e+00, 1.14818221e+03, 3.86046312e+02],
 [  0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

dist = np.array([[ -2.42565104e-01, -4.77893070e-02, -1.31388084e-03, -8.79107779e-05, 2.20573263e-02]])

img = cv2.imread('../Data/Frames_project_video/frame0.jpg')

undist = cv2.undistort(img, k, dist, None, k)

denoise = cv2.fastNlMeansDenoisingColored(undist,None,10,10,7,21)

gray = cv2.cvtColor(denoise, cv2.COLOR_BGR2GRAY)

gray_filtered = cv2.bilateralFilter(gray, 7, 50, 50)

# edges = cv2.Canny(gray, 60, 120)

ROI = gray_filtered[400 : gray_filtered.shape[0], 0 : gray_filtered.shape[1]]

# pts_src = np.load('points.npy')

# pts_dst = np.float32([(0, 0), (ROI.shape[1] - 1, 0), (0, ROI.shape[0] - 1), (ROI.shape[1] - 1, ROI.shape[0] - 1)])

# h, status = cv2.findHomography(pts_src, pts_dst)
    
# im_out = cv2.warpPerspective(ROI, h, (ROI.shape[1],ROI.shape[0]))

# Stacking the images to print them together for comparison
# images = np.hstack((edges, edges_filtered))


# f, ax = plt.subplots(1, 3, figsize=(12, 9))
# f.tight_layout()
# ax[0].imshow(img, cmap='gray')
# ax[1].imshow(undist, cmap='gray')
# ax[2].imshow(denoise, cmap='gray')
# plt.show()

cv2.imshow('src',  ROI)
# cv2.imshow('dst',  im_out)
# cv2.imshow('superimpose',  ROI)
cv2.imwrite('ROI.jpg', ROI)
while (1):
	k = cv2.waitKey(1) & 0xFF
	if k == 27:
		break

cv2.destroyAllWindows()