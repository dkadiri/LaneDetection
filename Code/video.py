import numpy as np
import cv2
import preprocess

cap = cv2.VideoCapture('../Data/project_video.mp4')
# frame_width = int(cap.get(3)) 
# frame_height = int(cap.get(4)) 
   
# size = (frame_width, frame_height)
# result = cv2.VideoWriter('result.avi',  
#                          cv2.VideoWriter_fourcc(*'MJPG'), 
#                          10, size)
while cap.isOpened():
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    
    k, dist = preprocess.cam_params()
    gray, hsv, mask_yellow, mask_white, combined_mask = preprocess.color_seg(frame, k, dist)
    im_out, invM = preprocess.BirdEyesViewUtils(combined_mask)
    left_index, right_index = preprocess.LaneCandidates(im_out)
    out_img = np.dstack((im_out, im_out, im_out))
    region = preprocess.region(out_img, left_index, right_index)
    cv2.fillConvexPoly(out_img, region, (200, 200, 0))
    ROI = frame[400 : frame.shape[0], 0 : frame.shape[1]]
    rev_im_out = cv2.warpPerspective(out_img, invM, (ROI.shape[1], ROI.shape[0]))
    frame_main = cv2.addWeighted(rev_im_out, 0.3, ROI, 0.7, 0)

    cv2.imshow('Main', frame_main)
    # result.write(frame_main)

    if cv2.waitKey(25) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()