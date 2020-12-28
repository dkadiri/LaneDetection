import cv2
import numpy as np
import numpy.polynomial.polynomial as poly
import matplotlib.pyplot as plt

k  = np.array([[  1.15422732e+03, 0.00000000e+00, 6.71627794e+02],
 [  0.00000000e+00, 1.14818221e+03, 3.86046312e+02],
 [  0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

dist = np.array([[ -2.42565104e-01, -4.77893070e-02, -1.31388084e-03, -8.79107779e-05, 2.20573263e-02]])

img = cv2.imread('../Data/Frames_project_video/frame0.jpg')

yellow_lower = np.array([20, 120, 120], dtype=np.uint8)
yellow_upper = np.array([40, 255, 255], dtype=np.uint8)

white_lower = np.array([0, 0, 225], dtype=np.uint8)
white_upper = np.array([255, 30, 255], dtype=np.uint8)

def step1(img, k, dist):
	undist = cv2.undistort(img, k, dist, None, k)
	denoise = cv2.fastNlMeansDenoisingColored(undist,None,10,10,7,21)
	gray = cv2.cvtColor(denoise, cv2.COLOR_BGR2GRAY)
	
	hsv = cv2.cvtColor(denoise,cv2.COLOR_BGR2HSV)
	mask_yellow = cv2.inRange(hsv, yellow_lower, yellow_upper)
	mask_white = cv2.inRange(hsv, white_lower, white_upper)
	combined_mask = cv2.bitwise_or(mask_yellow, mask_white)

	# yellow_output = cv2.bitwise_and(img1, img1, mask=mask_yellow1)

	# yellow_ratio =(cv2.countNonZero(mask_yellow1))/(img1.size/3)

	# print("Yellow in image", np.round(yellow_ratio*100, 2))
	return gray, hsv, mask_yellow, mask_white, combined_mask

gray, hsv, mask_yellow, mask_white, combined_mask = step1(img, k, dist)


# gray = np.hstack((gray1, gray2, gray3, gray4))
# gray = cv2.resize(gray, (1500, 200))

ROI = combined_mask[400 : combined_mask.shape[0], 0 : combined_mask.shape[1]]

pts_src = np.load('points.npy')
pts_src = np.float32(pts_src)
# print(pts_src)
pts_dst = np.float32([(0, 0), (200, 0), (0, 200), (200, 200)])

h, status = cv2.findHomography(pts_src, pts_dst)
    
im_out = cv2.warpPerspective(ROI, h, (200, 200))

# example = im_out[im_out.shape[0] // 2:, :]

hist = np.sum(im_out, axis=0)
midpoint = np.int(hist.shape[0] // 2)

left_top_regions = np.argwhere(hist[:midpoint]>10)
right_top_regions = np.argwhere(hist[midpoint:]>10) + midpoint

out_img = np.dstack((im_out, im_out, im_out))

left_index = []
right_index = []

for x in left_top_regions:
	for y in range(im_out.shape[0]):
		if im_out[y, x[0]] > 0:
			left_index.append((y, x[0]))

left_index = np.asarray(left_index)


for x in right_top_regions:
	for y in range(im_out.shape[0]):
		if im_out[y, x[0]] > 0:
			right_index.append((y, x[0]))

right_index = np.asarray(right_index)

left_x = left_index[::-1, 1]
left_y = left_index[:, 0]

f = np.poly1d(np.polyfit(left_x, left_y, 2))
t = np.linspace(np.min(left_x), np.max(left_x), 100)

plt.style.use('dark_background')
fig, ax = plt.subplots(1, 4, figsize=(20,6))

ax[0].imshow(ROI, cmap='gray')
# ax[0].axis("off")

ax[1].imshow(im_out, cmap='gray')
# ax[1].axis("off")

ax[2].plot(hist)

ax[3].imshow(out_img, cmap='gray')
ax[3].plot(left_index[:, 1], left_index[:, 0], '.', color='red')
ax[3].plot(right_index[:, 1], right_index[:, 0], '+', color='blue')
ax[3].plot(t, f(t), '-')

plt.show()


# cv2.imshow('ROI', ROI)
# cv2.imshow('Birds eye view', out_img)

# while (1):
# 	k = cv2.waitKey(1) & 0xFF
# 	if k == 27:
# 		break

# cv2.destroyAllWindows()