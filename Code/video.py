import numpy as np
import cv2
import preprocess
from tqdm import tqdm

cap = cv2.VideoCapture('../Data/challenge_video.mp4')
frame_width = int(cap.get(3)) 
frame_height = int(cap.get(4)) 
   
size = (frame_width, frame_height)
result = cv2.VideoWriter('challenge.avi',  
                         cv2.VideoWriter_fourcc(*'MJPG'), 
                         10, size)
# while cap.isOpened():
for i in tqdm(range(1249)):	
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    
    k, dist = preprocess.cam_params()
    denoise, gray, hsv, mask_yellow, mask_white, combined_mask = preprocess.color_seg(frame, k, dist)
    im_out, invM = preprocess.BirdEyesViewUtils(combined_mask)
    hist, left_index, right_index = preprocess.LaneCandidates(im_out)
    out_img = np.dstack((im_out, im_out, im_out))
    left_fitx, left_ploty, right_fitx, right_ploty, region = preprocess.region(out_img, left_index, right_index)
    cv2.fillConvexPoly(out_img, region, (200, 200, 0))
    rev_im_out = cv2.warpPerspective(out_img, invM, (frame.shape[1], frame.shape[0]))
    res = cv2.addWeighted(rev_im_out, 0.3, frame, 0.7, 0)

    # cv2.imshow('Main', res)
    result.write(res)

    if cv2.waitKey(25) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()