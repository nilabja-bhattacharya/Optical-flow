import numpy as np
import cv2 

def lucas_kadane(frame1, frame2, win = 2):
    
     # calculate gradients in x, y and t dimensions
    Ix = np.zeros(frame1.shape, dtype=np.float32)
    Iy = np.zeros(frame1.shape, dtype=np.float32)
    It = np.zeros(frame1.shape, dtype=np.float32)
    Ix[1:-1, 1:-1] = cv2.subtract(frame1[1:-1, 2:], frame1[1:-1, :-2]) / 2
    Iy[1:-1, 1:-1] = cv2.subtract(frame1[2:, 1:-1], frame1[:-2, 1:-1]) / 2
    It[1:-1, 1:-1] = cv2.subtract(frame1[1:-1, 1:-1], frame2[1:-1, 1:-1])

    params = np.zeros(frame1.shape + (5,))
    params[..., 0] = Ix ** 2
    params[..., 1] = Iy ** 2
    params[..., 2] = Ix * Iy
    params[..., 3] = Ix * It
    params[..., 4] = Iy * It
    del It, Ix, Iy
    cum_params = np.cumsum(np.cumsum(params, axis=0), axis=1)
    del params
    win_params = (cum_params[2 * win + 1:, 2 * win + 1:] -
                  cum_params[2 * win + 1:, :-1 - 2 * win] -
                  cum_params[:-1 - 2 * win, 2 * win + 1:] +
                  cum_params[:-1 - 2 * win, :-1 - 2 * win])
    del cum_params
    op_flow = np.zeros(frame1.shape + (2,))
    det = win_params[...,0] * win_params[..., 1] - win_params[..., 2] **2

    op_flow_x = np.where(det != 0,
                         (win_params[..., 1] * win_params[..., 3] -
                          win_params[..., 2] * win_params[..., 4]) / det,
                         0)
    op_flow_y = np.where(det != 0,
                         (win_params[..., 0] * win_params[..., 4] -
                          win_params[..., 2] * win_params[..., 3]) / det,
                         0)
    op_flow[win + 1: -1 - win, win + 1: -1 - win, 0] = op_flow_x[:-1, :-1]
    op_flow[win + 1: -1 - win, win + 1: -1 - win, 1] = op_flow_y[:-1, :-1]
    op_flow = op_flow.astype(np.float32)
    return op_flow