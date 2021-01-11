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
	
	hsv = cv2.cvtColor(denoise,cv2.COLOR_BGR2HSV)
	mask_yellow = cv2.inRange(hsv, yellow_lower, yellow_upper)
	mask_white = cv2.inRange(hsv, white_lower, white_upper)
	combined_mask = cv2.bitwise_or(mask_yellow, mask_white)

	# yellow_output = cv2.bitwise_and(img1, img1, mask=mask_yellow1)

	# yellow_ratio =(cv2.countNonZero(mask_yellow1))/(img1.size/3)

	# print("Yellow in image", np.round(yellow_ratio*100, 2))
	return combined_mask

def BirdEyesViewUtils(img):
	pts_src = np.load('points.npy')
	pts_src = np.float32(pts_src)
	pts_dst = np.float32([(0, 0), (img.shape[1], 0), (img.shape[1], img.shape[0]), (0, img.shape[0])])

	M = cv2.getPerspectiveTransform(pts_src, pts_dst)
	invM = cv2.getPerspectiveTransform(pts_dst, pts_src)

	im_out = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))
	return im_out, invM

def LaneCandidates(im_out):
	hist = np.sum(im_out[im_out.shape[0] // 2:, :], axis=0)
	out_img = np.dstack((im_out, im_out, im_out)) * 255
	midpoint = np.int(hist.shape[0] // 2)

	left_x_base = np.argmax(hist[:midpoint])
	right_x_base = np.argmax(hist[midpoint:]) + midpoint

	windows = 12
	margin = 100 
	min_pixels = 50

	window_ht = np.int(im_out.shape[0] // windows)

	nonZero = im_out.nonzero()
	nonZero_y = np.array(nonZero[0])
	nonZero_x = np.array(nonZero[1])
	
	current_left_x, current_right_x = left_x_base, right_x_base
	left_lane_index, right_lane_index = [], []

	for window in range(windows):
		window_y_low = im_out.shape[0] - (window + 1) * window_ht
		window_y_high = im_out.shape[0] - window * window_ht
		left_window_x_low = current_left_x - margin
		left_window_x_high = current_left_x + margin
		right_window_x_low = current_right_x - margin
		right_window_x_high = current_right_x + margin
		cv2.rectangle(out_img, (left_window_x_low, window_y_low), (left_window_x_high, window_y_high), (0, 255, 0), 4)
		cv2.rectangle(out_img, (right_window_x_low, window_y_low), (right_window_x_high, window_y_high), (0, 255, 0), 4)
		good_left_inds = ((nonZero_y >= window_y_low) & (nonZero_y < window_y_high) & (nonZero_x >= left_window_x_low) & (nonZero_x < left_window_x_high)).nonzero()[0]
		good_right_inds = ((nonZero_y >= window_y_low) & (nonZero_y < window_y_high) & (nonZero_x >= right_window_x_low) & (nonZero_x < right_window_x_high)).nonzero()[0]
		left_lane_index.append(good_left_inds)
		right_lane_index.append(good_right_inds)
		if len(good_left_inds) > min_pixels:
			current_left_x = np.int(np.mean(nonZero_x[good_left_inds]))
		if len(good_right_inds) > min_pixels:
			current_right_x = np.int(np.mean(nonZero_x[good_right_inds]))
	
	left_lane_index = np.concatenate(left_lane_index)
	right_lane_index = np.concatenate(right_lane_index)


	left_x = nonZero_x[left_lane_index]
	left_y = nonZero_y[left_lane_index]
	right_x = nonZero_x[right_lane_index]
	right_y = nonZero_y[right_lane_index]

	left_index = (left_x, left_y)
	right_index = (right_x, right_y)

	return left_index, right_index


def poly_fit(img, indx):
	x, y = indx
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

	return region