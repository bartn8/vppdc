from dgp.datasets import SynchronizedSceneDataset
import numpy as np
import cv2
import os
import argparse
import tqdm
import random
from numba import njit

@njit
def conti_conf(dmap, n=7, th=15):
    """
    Return a confidence map based on Conti's method (https://arxiv.org/abs/2210.03118).
    Points in a window that are far from foreground are rejected.
    Parameters
    ----------
    dmap: HxW np.ndarray
        Depth map used to extract confidence map.
    n: int
        Window size (3,5,7,...)
    th: float
        Threshold for absolute difference
    Returns
    -------
    conf_rst: HxW np.ndarray
        Binary confidence map (1 for rejected points)
    """
    h,w = dmap.shape[:2] 

    #Confidence map between 0 and 1 (binary)
    conf_map = np.zeros(dmap.shape, dtype=np.uint8)

    n = n//2
    
    #Conti's filtering method
    for y in range(h):
        for x in range(w):
            if dmap[y,x] > 0:
                #Search min
                dmin = 1000000
                for yw in range(-n,n+1):
                    for xw in range(-n,n+1):
                        if 0 <= y+yw and y+yw <= h-1 and 0 <= x+xw and x+xw <= w-1:
                            if dmap[y+yw,x+xw] < dmin and dmap[y+yw,x+xw] > 1e-3:
                                dmin = dmap[y+yw,x+xw]

                #Find pixel-wise confidence
                for yw in range(-n,n+1):
                    for xw in range(-n,n+1):
                        if 0 <= y+yw and y+yw <= h-1 and 0 <= x+xw and x+xw <= w-1:
                            #Absolute thresholding
                            if dmap[y+yw,x+xw]-dmin > th:
                                conf_map[y+yw,x+xw] = 1
            else:
                conf_map[y,x] = 1

    return conf_map

@njit
def filter(dmap,conf_map,th):
    """
    Drop points from a disparity map based on a confidence map.
    
    Parameters
    ----------
    dmap: HxW np.ndarray
        Disparity map to modify: there is side-effect.    
    conf_map: HxW np.ndarray
        Confidence map to use for filtering (1 if point is filtered).
    th: float
        Threshold for filtering

    Returns
    -------
    filtered_i: int
        Number of points filtered
    """
    h,w = dmap.shape[:2]
    filtered_i = 0
    for y in range(h):
        for x in range(w):
            if dmap[y,x] > 0:
                if conf_map[y,x] > th:
                    dmap[y,x] = 0
                    filtered_i += 1
    return filtered_i

def filter_heuristic(dmap, n=9, th=1, th_filter=0.1):
    filtered_dmap = dmap.copy()
    conf_map = conti_conf(dmap, n, th)
    _ = filter(filtered_dmap, conf_map, th_filter)
    return filtered_dmap

def sample_hints(hints, probability=0.20):
    validhints = hints > 0
    if probability < 1:
        new_validhints = (validhints * (np.random.rand(*validhints.shape) < probability))
        new_hints = hints * new_validhints  # zero invalid hints
        new_hints[new_validhints==0] = 0
    else:
        new_hints = hints
        new_validhints = validhints
    #new_hints[new_hints>5000] = 0
    return new_hints

parser = argparse.ArgumentParser(description='DDAD Converter')
parser.add_argument('--input_path', '-i', required=True, type=str, help='DDAD folder path (it contains ddad.json file)')
parser.add_argument('--output_path', '-o', required=True, type=str, help='Converted DDAD folder path')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)

dataset = SynchronizedSceneDataset(f'{args.input_path}/ddad.json',
    datum_names=('camera_01', 'lidar'),
    generate_depth_from_datum='lidar',
    split='val')
   

basepath = args.output_path
savepath = os.path.join(basepath, "val")

savepath_rgb = os.path.join(savepath, "rgb")
savepath_hints = os.path.join(savepath, "hints")
savepath_gt = os.path.join(savepath, "gt")
savepath_k = os.path.join(savepath, "intrinsics")

for path in [basepath, savepath, savepath_rgb, savepath_hints, savepath_gt, savepath_k]:
    if not os.path.exists(path):
        try:
            os.makedirs(path)
            print(f"Directory created: {path}")
        except OSError as e:
            print(f"Error creating directory {path}: {e}")
    else:
        print(f"Directory already exists: {path}")

k=0
for i in tqdm.tqdm(range(len(dataset))):
    camera_01 = dataset[i][0][0]
    rgb = camera_01['rgb']# PIL.Image
    depth = camera_01['depth']# (H,W) numpy.ndarray, generated from 'lidar'
    K = camera_01['intrinsics']

    sparse_lidar = depth.copy()
    
    # Gaussian sampling
    sparse_lidar = sample_hints(sparse_lidar, 0.2)

    #Remove occlusion in depth
    depth = filter_heuristic(depth, 7, 3)

    #kitti format conversion
    depth = (depth*256.0).astype(np.uint16)
    sparse_lidar = (sparse_lidar*256.0).astype(np.uint16)
    bgr = cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)
    
    #save to disk
    cv2.imwrite(os.path.join(savepath_rgb, f"{k:010d}.png"), bgr)
    cv2.imwrite(os.path.join(savepath_hints, f"{k:010d}.png"), sparse_lidar)
    cv2.imwrite(os.path.join(savepath_gt, f"{k:010d}.png"), depth)
    np.savetxt(os.path.join(savepath_k, f"{k:010d}.txt"), K)
    k += 1
    
    