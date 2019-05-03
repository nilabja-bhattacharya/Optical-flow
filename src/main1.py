import numpy as np
import cv2
import time
import Lk



help_message = \
    '''
USAGE: optical_flow.py [<video_source>]
Keys:
 1 - To save tracked moving objects image
 2 - To save segmented image
 3 - To save optical flow image
'''
count = 0

cv2.namedWindow('hsv',cv2.WINDOW_NORMAL)
cv2.namedWindow('Image',cv2.WINDOW_NORMAL)
cv2.namedWindow('flow',cv2.WINDOW_NORMAL)
def draw_flow(img, flow, step=16):
    
    #from the beginning to position 2 (excluded channel info at position 3)
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y,x].T
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(vis, lines, 0, (0, 255, 0))
    for (x1, y1), (x2, y2) in lines:
        cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    return vis


def draw_hsv(flow, img):
    hsv = np.zeros_like(img)
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,1] = 255
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    cv2.imshow('hsv', bgr)
    return bgr


def warp_flow(img, flow):
    (h, w) = flow.shape[:2]
    flow = -flow
    flow[:, :, 0] += np.arange(w)
    flow[:, :, 1] += np.arange(h)[:, np.newaxis]
    res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
    return res


if __name__ == '__main__':
    import sys
    print (help_message)
    try:
        fn = sys.argv[1]
    except:
        fn = 0

    cam = cv2.VideoCapture(fn)
    (ret, prev) = cam.read()

    prevgray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    show_hsv = True
    idx = 0
    while True:
        idx +=1
        (ret, img) = cam.read()
        vis = img.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        flow = Lk.lucas_kadane(prevgray, gray, 15)
        #flow = cv2.calcOpticalFlowFarneback(prevgray,gray,None,0.5,5,15,3,5,1.1,cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
        prevgray = gray
        fl = draw_flow(gray, flow)
        cv2.imshow('flow', fl)
        if show_hsv:
            hsv1 = draw_hsv(flow, prev)
            gray1 = cv2.cvtColor(hsv1, cv2.COLOR_BGR2GRAY)
            thresh = cv2.threshold(gray1, 25, 0xFF,
                                   cv2.THRESH_BINARY)[1]
            thresh = cv2.dilate(thresh, None, iterations=2)
            gray2, cnts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # loop over the contours
            for c in cnts:

                # if the contour is too small, ignore it
                (x, y, w, h) = cv2.boundingRect(c)
                if w > 15 and h > 15 and w < 900 and h < 680:
                    cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 0xFF, 0), 4)
                    #cv2.putText(vis,str(time.time()),(x, y),cv2.FONT_HERSHEY_SIMPLEX,1,(0, 0, 0xFF),1)
            cv2.imshow('Image', vis)
        ch = 0xFF & cv2.waitKey(5)
        if ch == 27:
            break
        if ch == ord('1'):
            cv2.imwrite("output/imagetrack/imagetrack"+str(idx)+'.png', vis)
        if ch == ord('2'):
            cv2.imwrite("output/imagehsv/imagehsv"+str(idx)+'.png', hsv1)
        if ch == ord('3'):
            cv2.imwrite("output/imageflow/imageflow"+str(idx)+'.png', fl)
    cv2.destroyAllWindows()


			
