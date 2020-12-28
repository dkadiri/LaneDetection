import numpy as np
import cv2

ix,iy = -1,-1
# mouse callback function
def draw_circle(event,x,y,flags,param):
    global ix,iy
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(img,(x,y),5,(255,0,0),-1)
        ix,iy = x,y

# Create a black image, a window and bind the function to window
img = cv2.imread('ROI.jpg')
cv2.namedWindow('image')
cv2.setMouseCallback('image',draw_circle)

points = []

while(1):
    cv2.imshow('image',img)
    k = cv2.waitKey(20) & 0xFF
    if k == 27:
        break
    elif k == ord('a'):
        points.append((ix, iy))

np.save('points.npy', points)
# d = np.load('points.npy')
print(points)
cv2.destroyAllWindows()
