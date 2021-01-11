import cv2
import numpy as np
import numpy.polynomial.polynomial as poly
import matplotlib.pyplot as plt

def cam_params():
	k  = np.array([[  1.15422732e+03, 0.00000000e+00, 6.71627794e+02],
	 [  0.00000000e+00, 1.14818221e+03, 3.86046312e+02],
	 [  0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

	dist = np.array([[ -2.42565104e-01, -4.77893070e-02, -1.31388084e-03, -8.79107779e-05, 2.20573263e-02]])

	return k, dist

def color_seg(img, k, dist):
	yellow_lower = np.array([20, 100, 100], dtype=np.uint8)
	yellow_upper = np.array([40, 255, 255], dtype=np.uint8)

	white_lower = np.array([0, 0, 215], dtype=np.uint8)
	white_upper = np.array([255, 40, 255], dtype=np.uint8)
	
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
	return denoise, gray, hsv, mask_yellow, mask_white, combined_mask

def BirdEyesViewUtils(img):
	# ROI = combined_mask[400 : combined_mask.shape[0], 0 : combined_mask.shape[1]]

	pts_src = np.load('points.npy')
	pts_src = np.float32(pts_src)
	pts_dst = np.float32([(0, 0), (img.shape[1], 0), (img.shape[1], img.shape[0]), (0, img.shape[0])])

	M = cv2.getPerspectiveTransform(pts_src, pts_dst)
	invM = cv2.getPerspectiveTransform(pts_dst, pts_src)

	im_out = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))
	# im_out = cv2.cvtColor(im_out, cv2.COLOR_BGR2GRAY)
	return im_out, invM

def top_regions(left, im_out):
	index = []

	for x in left:
		for y in range(im_out.shape[0]):
			if im_out[y, x[0]] > 0:
				index.append((y, x[0]))

	index = np.asarray(index)

	return index


def LaneCandidates(im_out):
	# im_out = cv2.cvtColor(im_out, cv2.COLOR_BGR2GRAY)
	hist = np.sum(im_out, axis=0)
	midpoint = np.int(hist.shape[0] // 2)

	left_top_regions = np.argwhere(hist[:midpoint]>10)
	right_top_regions = np.argwhere(hist[midpoint:]>10) + midpoint

	left_index = top_regions(left_top_regions, im_out)
	right_index = top_regions(right_top_regions, im_out)

	# out_img = np.dstack((im_out, im_out, im_out))

	# left_index = []
	# right_index = []

	# for x in left_top_regions:
	# 	for y in range(im_out.shape[0]):
	# 		if im_out[y, x[0]] > 0:
	# 			left_index.append((y, x[0]))

	# left_index = np.asarray(left_index)

	# right_index = []

	# for x in right_top_regions:
	# 	for y in range(im_out.shape[0]):
	# 		if im_out[y, x[0]] > 0:
	# 			right_index.append((y, x[0]))

	# right_index = np.asarray(right_index)
	return hist, left_index, right_index


def poly_fit(img, points):
	x = points[:, 1]
	y = points[:, 0]
	coefs = np.polyfit(y, x, 2)
	ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])
	fitx = coefs[0] * ploty ** 2 + coefs[1] * ploty + coefs[2]
	return fitx, ploty

def region(out_img, left_index, right_index):
	left_fitx, left_ploty = poly_fit(out_img, left_index)
	right_fitx, right_ploty = poly_fit(out_img, right_index)

	left_fitx, left_ploty = np.int32(left_fitx), np.int32(left_ploty)
	right_fitx, right_ploty,  = np.int32(right_fitx), np.int32(right_ploty)

	region = np.array([[left_fitx[0], left_ploty[0]], [left_fitx[-1], left_ploty[-1]], [right_fitx[-1], right_ploty[-1]], [right_fitx[0], right_ploty[0]]])

	return left_fitx, left_ploty, right_fitx, right_ploty, region

k, dist = cam_params()

img = cv2.imread('../Data/Frames_project_video/frame1036.jpg')

denoise, gray, hsv, mask_yellow, mask_white, combined_mask = color_seg(img, k, dist)

im_out, invM = BirdEyesViewUtils(combined_mask)

# hist, left_index, right_index = LaneCandidates(im_out)

# out_img = np.dstack((im_out, im_out, im_out))

# left_fitx, left_ploty, right_fitx, right_ploty, region = region(out_img, left_index, right_index)

# cv2.fillConvexPoly(out_img, region, (200, 200, 0))

#ROI = img[400 : img.shape[0], 0 : img.shape[1]]

# rev_im_out = cv2.warpPerspective(out_img, invM, (img.shape[1], img.shape[0]))

# frame = cv2.addWeighted(rev_im_out, 0.3, img, 0.7, 0)

# plt.style.use('dark_background')
# fig, ax = plt.subplots(1, 6, figsize=(20, 2), constrained_layout=False)
# fig.tight_layout()

# ax[0].imshow(cv2.cvtColor(denoise, cv2.COLOR_BGR2RGB), aspect="auto")
# ax[0].set(title='Undistorted & Denoised')
# plt.setp(ax[0].get_xticklabels(), visible=False)
# plt.setp(ax[0].get_yticklabels(), visible=False)
# ax[0].tick_params(axis='both', which='both', length=0)

# ax[1].imshow(combined_mask, cmap='gray', aspect="auto")
# ax[1].set(title='Color Thresholding')
# plt.setp(ax[1].get_xticklabels(), visible=False)
# plt.setp(ax[1].get_yticklabels(), visible=False)
# ax[1].tick_params(axis='both', which='both', length=0)

# ax[2].imshow(im_out, cmap='gray', aspect="auto")
# ax[2].set(title='BirdEyesView')
# plt.setp(ax[2].get_xticklabels(), visible=False)
# plt.setp(ax[2].get_yticklabels(), visible=False)
# ax[2].tick_params(axis='both', which='both', length=0)

# ax[3].plot(hist)
# ax[3].set(title='Histogram along Y-axis')
# ax[3].set_aspect('auto')

# ax[4].imshow(out_img, cmap='gray', aspect="auto")
# ax[4].set(title='Lane Pixel Candidates')
# ax[4].plot(left_index[:, 1], left_index[:, 0], '.', color='red',  markersize=3)
# ax[4].plot(right_index[:, 1], right_index[:, 0], '+', color='blue',  markersize=3)
# plt.setp(ax[4].get_xticklabels(), visible=False)
# plt.setp(ax[4].get_yticklabels(), visible=False)
# ax[4].tick_params(axis='both', which='both', length=0)

# ax[5].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), aspect="auto")
# ax[5].set(title='Backprojection')
# # ax[5].plot(left_index[:, 1], left_index[:, 0], '.', color='red',  markersize=3)
# # ax[5].plot(right_index[:, 1], right_index[:, 0], '+', color='blue',  markersize=3)
# # ax[5].plot(left_fitx, left_ploty, 'y-')
# # ax[5].plot(right_fitx, right_ploty, 'y-')
# # ax[5].plot(left_fitx[0], left_ploty[0], 'b*', left_fitx[-1], left_ploty[-1], 'bo', markersize=5)
# plt.setp(ax[5].get_xticklabels(), visible=False)
# plt.setp(ax[5].get_yticklabels(), visible=False)
# ax[5].tick_params(axis='both', which='both', length=0)
# # ax[5].plot(rev_im_out, cmap='gray')
# plt.savefig('Pipeline.jpg', dpi=150)
# plt.show()

# for i in left_index:
# 	cv2.circle(out_img, (i[1], i[0]), 3, (0, 0, 255), -1)

# for i in right_index:
# 	cv2.circle(out_img, (i[1], i[0]), 3, (255, 0, 0), -1)

# cv2.imshow('Main frame', img)
cv2.imshow('ROI', combined_mask)
cv2.imwrite('Undistorted.jpg', denoise)
cv2.imwrite('combined_mask.jpg', combined_mask)
cv2.imwrite('topview.jpg', im_out)

# cv2.imshow('Yellow', hist)
# cv2.imshow('Inverse', rev_im_out)

while (1):
	k = cv2.waitKey(1) & 0xFF
	if k == 27:
		break

cv2.destroyAllWindows()