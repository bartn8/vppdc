import cv2
import numpy as np
import shutil
import os
from numba import njit

def sgm_opencv(imgL, imgR, maxdisp=192, p_factor=7, w_size=16):
    # disparity range is tuned for 'aloe' image pair
    window_size = p_factor
    min_disp = 0
    num_disp = maxdisp - min_disp
    stereo = cv2.StereoSGBM_create(minDisparity=min_disp,
                                   numDisparities=num_disp,
                                   blockSize=w_size,
                                   P1=8 * 3 * window_size ** 2,
                                   P2=32 * 3 * window_size ** 2,
                                   disp12MaxDiff=3,
                                   uniquenessRatio=10,
                                   speckleWindowSize=150,
                                   speckleRange=32
                                   )

    # print('computing disparity...')
    padded_left = cv2.copyMakeBorder(imgL, 0,0,num_disp,0,cv2.BORDER_CONSTANT,None,0)
    padded_right = cv2.copyMakeBorder(imgR, 0,0,num_disp,0,cv2.BORDER_CONSTANT,None,0)
    disp = stereo.compute(padded_left, padded_right).astype(np.float32) / 16.0
    disp = disp[:,num_disp:]
    _interpolate_background(disp)
    return disp

def backup_source_code(backup_directory):
    ignore_hidden = shutil.ignore_patterns(
        ".", "..", ".git*", "*pycache*", "*build", "*.fuse*", "*_drive_*",
        "*pretrained*", "*log*", "*.vscode*", "*tmp*", "*weights*", "*thirdparty*")

    if os.path.exists(backup_directory):
        shutil.rmtree(backup_directory)

    shutil.copytree('.', backup_directory, ignore=ignore_hidden)
    os.system("chmod -R g+w {}".format(backup_directory))

#https://github.com/simonmeister/motion-rcnn/blob/master/devkit/cpp/io_disp.h
@njit
def _interpolate_background(dmap):
    h,w = dmap.shape[:2]

    for v in range(h):
        count = 0
        for u in range(w):
            if dmap[v,u] > 0:
                if count >= 1:#at least one pixel requires interpolation
                    u1,u2 = u-count,u-1#first and last value for interpolation
                    if u1>0 and u2<w-1:#set pixel to min disparity
                        d_ipol = min(dmap[v,u1-1], dmap[v,u2+1])
                        for u_curr in range(u1,u2+1):
                            dmap[v,u_curr] = d_ipol
                count = 0
            else:
                count +=1
        
        #Border interpolation(left,right): first valid dmap value is used as filler
        for u in range(w):
            if dmap[v,u] > 0:
                for u2 in range(u):
                    dmap[v,u2] = dmap[v,u]
                break

        for u in range(w-1,-1,-1):
            if dmap[v,u] > 0:
                for u2 in range(u+1,w):
                    dmap[v,u2] = dmap[v,u]
                break
        
    #Border interpolation(top,bottom): first valid dmap value is used as filler
    for u in range(w):
        for v in range(h):
            if dmap[v,u] > 0:
                for v2 in range(v):
                    dmap[v2,u] = dmap[v,u]
                break
        
        for v in range(h-1,-1,-1):
            if dmap[v,u] > 0:
                for v2 in range(v+1,h):
                    dmap[v2,u] = dmap[v,u]
                break

