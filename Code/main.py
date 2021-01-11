import numpy as np
import cv2
import preprocess
from tqdm import tqdm

cap = cv2.VideoCapture('../Data/project_video.mp4')
frame_width = int(cap.get(3)) 
frame_height = int(cap.get(4)) 
   
size = (frame_width, frame_height)
result = cv2.VideoWriter('../Results/result.avi',  
                         cv2.VideoWriter_fourcc(*'MJPG'), 
                         10, size)
k, dist = preprocess.cam_params()

# while cap.isOpened():
for i in tqdm(range(1249)):	
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    combined_mask = preprocess.color_seg(frame, k, dist)
    im_out, invM = preprocess.BirdEyesViewUtils(combined_mask)
    left_index, right_index = preprocess.LaneCandidates(im_out)
    region_img = np.dstack((im_out, im_out, im_out))
    region = preprocess.region(region_img, left_index, right_index)
    cv2.fillConvexPoly(region_img, region, (200, 200, 0))
    rev_im_out = cv2.warpPerspective(region_img, invM, (frame.shape[1], frame.shape[0]))
    res = cv2.addWeighted(rev_im_out, 0.3, frame, 0.7, 0)

    # cv2.imshow('Main', res)
    result.write(res)

    if cv2.waitKey(25) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()