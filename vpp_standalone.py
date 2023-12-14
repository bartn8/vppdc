import numpy as np
import cv2
import math
from numba import njit


@njit
def random_int_256():
    return np.random.randint(0, 256)

@njit
def random_int_max(n):
    return np.random.randint(0, n)

@njit
def _virtual_projection_scan_rnd(ctx, l, r, g, filled_g, uniform_color, wsize, direction, c, c_occ, g_occ, discard_occ, interpolate, use_context):

    """
    Virtual projection using sparse disparity.

    Parameters
    ----------
    ctx: np.numpy [H,W,C] np.uint8
        Context image
    l: np.numpy [H,W,C] np.uint8
        Left original image
    r: np.numpy [H,W,C] np.uint8
        Right original image        
    g: np.numpy [H,W] np.float32
        Sparse disparity
    filled_g: np.numpy [H,W] np.uint8
        Filled sparse disparity
    uniform_color: bool
        Patches have same color (true)
    wsize: int
        Max projection patch size (Default 5)                          
    direction: int mod 2
        Projection direction (left->right or right->left) (Default 0)
    c: float
        alpha blending factor
    c_occ: float
        alpha blending factor in occluded areas
    g_occ: np.numpy [H,W] np.uint8
        Occlusion mask (If not present use np.zeros(l.shape, dtype=np.uint8))
    discard_occ: bool
        Discard occluded points
    interpolate: bool
        sub-pixel disparities splatting
    use_context: bool
        Use context colors for patches (with exception of central point)


    Returns
    -------
    sample_i:
        number of points projected
    """

    sample_i = 0

    height, width, channels = l.shape[:3]
    
    #Window size factor (wsize=2*n+1 -> n = (wsize-1)/2)
    n = ((wsize -1) // 2)
    
    for y in range(height):
        x = width-1 if direction == 0 else 0
        while (direction != 0 and x < width) or (direction == 0 and x>=0):
            if g[y,x] > 0:
                d = round(g[y,x])
                d0 = math.floor(g[y,x]) #5.43 -> 5
                d1 = math.ceil(g[y,x])  #5.43 -> 6
                d1_blending = g[y,x]-d0   #0.43 -> d_blending = 1-0.43 = 0.57

                #Warping right (negative disparity hardcoded)
                xd = x-d
                xd0 = x-d0
                xd1 = x-d1
                
                for j in range(channels):

                    #1)Pattern color part
                    #Search for the best color to blend in the image
                    if uniform_color:
                        tmp_rvalue = None
                        rvalue = random_int_256()

                    min_n_y, max_n_y, min_n_x, max_n_x =  n,n,n,n

                    #Project patch in left and right images (1)
                    #Also in left side occlusion maintain uniformity (2)

                    for yw in range(-min_n_y,max_n_y+1):
                        for xw in range(-min_n_x,max_n_x+1):
                            if 0 <= y+yw and y+yw <= height-1 and 0 <= x+xw and x+xw <= width-1: 
                                if filled_g[y+yw,x+xw] > 0:
                                    
                                    #1)Pattern color part
                                    #Search for the best color to blend in the image
                                    if not uniform_color:
                                        tmp_rvalue = None
                                        rvalue = random_int_256()


                                    if use_context:
                                        if yw != 0 or xw != 0:
                                            if g[y+yw,x+xw] <= 0:
                                                if tmp_rvalue is None:
                                                    tmp_rvalue = rvalue
                                                rvalue = ctx[y+yw,x+xw,j]
                                            elif tmp_rvalue is not None:
                                                rvalue = tmp_rvalue
                                                tmp_rvalue = None
                                        # rvalue = ctx[y+yw,x+xw,j]

                                    if  0 <= xd0+xw and xd0+xw <= width-1:#  (1)
                                        #Occlusion check
                                        if g_occ[y,x] == 0:#Not occluded point  
                                            l[y+yw,x+xw,j] = (rvalue * c + l[y+yw,x+xw,j] * (1-c))
                                            if interpolate:
                                                r[y+yw,xd0+xw,j] = (((rvalue * c + r[y+yw,xd0+xw,j] * (1-c)) * (1-d1_blending)) + r[y+yw,xd0+xw,j] * d1_blending)
                                                if 0 <= xd1+xw and xd1+xw <= width-1:# Linear interpolation only if inside the border
                                                    r[y+yw,xd1+xw,j] = (((rvalue * c + r[y+yw,xd1+xw,j] * (1-c)) * d1_blending) + r[y+yw,xd1+xw,j] * (1-d1_blending))
                                            else:
                                                r[y+yw,xd+xw,j] = rvalue * c + r[y+yw,xd+xw,j] * (1-c)
                                        elif not discard_occ:# Occluded point: Foreground point should be projected before occluded point
                                            if interpolate:
                                                r[y+yw,xd0+xw,j] = (((rvalue * c_occ + r[y+yw,xd0+xw,j] * (1-c_occ)) * (1-d1_blending)) + r[y+yw,xd0+xw,j] * d1_blending)
                                                if 0 <= xd1+xw and xd1+xw <= width-1:
                                                    r[y+yw,xd1+xw,j] = (((rvalue * c_occ + r[y+yw,xd1+xw,j] * (1-c_occ)) * d1_blending) + r[y+yw,xd1+xw,j] * (1-d1_blending))
                                                l[y+yw,x+xw,j] = ((r[y+yw,xd0+xw,j]*(1-d1_blending)+r[y+yw,xd1+xw,j]*d1_blending) * c + l[y+yw,x+xw,j] * (1-c))             
                                            else:
                                                r[y+yw,xd+xw,j] = rvalue * c_occ + r[y+yw,xd+xw,j] * (1-c_occ)
                                                l[y+yw,x+xw,j] = r[y+yw,xd+xw,j] * c + l[y+yw,x+xw,j] * (1-c)
                                                
                                    else:#Left side occlusion (known) (2)
                                        l[y+yw,x+xw,j] = (rvalue * c + l[y+yw,x+xw,j] * (1-c))        
                                                    
                sample_i +=1

            x = x-1 if direction == 0 else x+1
    
    return sample_i

@njit
def _bilateral_filling(dmap, img, n, o_xy = 2, o_i= 1, th=.001):
    h,w = img.shape[:2]
    assert dmap.shape == img.shape
    cmap = np.zeros_like(dmap)
    aug_dmap = dmap.copy()

    for y in range(h):
        for x in range(w):
            i_ref = img[y,x]
            d_ref = dmap[y,x]
            if d_ref > 0:
                for yw in range(-n,n+1):
                    for xw in range(-n,n+1):
                        if 0 <= y+yw and y+yw <= h-1 and 0 <= x+xw and x+xw <= w-1:
                            weight = math.exp(-(((yw)**2+(xw)**2)/(2*(o_xy**2)) + ((img[y+yw,x+xw]-i_ref)**2)/(2*(o_i**2))))
                            if cmap[y+yw,x+xw] < weight:
                                cmap[y+yw,x+xw] = weight
                                aug_dmap[y+yw,x+xw] = d_ref
    
    aug_dmap = np.where(cmap>th,aug_dmap,0)

    return aug_dmap
       

def vpp(context, left, right, gt, wsize = 3,
        left2right = True, blending = 1.0, uniform_color = False, method="rnd",
        o_xy = 2, o_i= 1, fillingThreshold = 0.001, useFilling = True, useContext = False,
        c_occ = 0.00, g_occ = None, discard_occ = False,
        interpolate = True):
    
    lc,rc = np.copy(left), np.copy(right)
    gt = gt.astype(np.float32)

    assert method in ["rnd"]
    direction = 1 if left2right else 0

    if len(lc.shape) < 3:
        lc,rc = np.expand_dims(lc, axis=-1), np.expand_dims(rc, axis=-1)
    
    #Convert rgb to gray if needed
    if len(context.shape) == 3 and context.shape[2] == 3:
        gray_context = cv2.cvtColor(context, cv2.COLOR_BGR2GRAY)
    else:
        gray_context = np.squeeze(context)

    if useFilling:
        filled_g = _bilateral_filling(gt, gray_context, (wsize-1)//2, o_xy, o_i, th=fillingThreshold)
    else:
        filled_g = np.ones_like(gt)

    if g_occ is None:
        g_occ = np.zeros_like(gt)        

    if method == "rnd":
        _virtual_projection_scan_rnd(context,lc,rc,gt, filled_g, uniform_color, wsize, direction, blending, c_occ, g_occ, discard_occ,interpolate, useContext)
    return lc,rc
