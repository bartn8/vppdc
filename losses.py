
import torch
import numpy as np

def sample_hints(hints, validhints, probability=0.20):
    if probability < 1:
        new_validhints = (validhints * (torch.rand_like(validhints, dtype=torch.float32) < probability)).float()
        new_hints = hints * new_validhints  # zero invalid hints
        new_hints[new_validhints==0] = 0
    else:
        new_hints = hints
        new_validhints = validhints
    #new_hints[new_hints>5000] = 0
    return new_hints, new_validhints

def kitti_depth_metrics(disp, gt, valid, baseline = None, focal = None, max_depth = None, clip_depth = None):
    if len(disp.shape) == 4:
        disp = disp.squeeze(0)
    if len(valid.shape) == 4:
        valid = valid.squeeze(0)
    if len(gt.shape) == 4:
        gt = gt.squeeze(0)

    #conversion to depth
    if baseline is not None and focal is not None:
        depth = disp.copy()
        depth[disp>0] = ((focal*baseline)/disp)[disp>0]
        gt_depth = gt.copy()
        gt_depth[gt>0] = ((focal*baseline)/gt)[gt>0]
    else:
        depth = disp
        gt_depth = gt

    if max_depth is not None:
        if max_depth > 0:
            valid = np.where((valid > 0) & (gt <= max_depth),1,0).astype(np.uint8)

    if clip_depth is not None:
        if clip_depth > 0:
            disp[disp > clip_depth] = clip_depth

    error = (depth[valid>0]-gt_depth[valid>0])
    
    inv_depth = depth.copy()
    inv_depth[depth>0] = 1.0 / depth[depth>0]
    inv_gt_deth = gt_depth.copy()
    inv_gt_deth[gt_depth>0] = 1.0 / gt_depth[gt_depth>0]

    inv_error = np.abs(inv_depth[valid>0] - inv_gt_deth[valid>0])

    mae = np.abs(error).mean()
    rmse = np.sqrt((error ** 2).mean())
    imae = inv_error.mean()
    irmse = np.sqrt((inv_error ** 2).mean())
    rel = np.abs(error/ gt_depth[valid>0]).mean()


    return {'mae': mae, 'rmse': rmse, 'imae': imae, 'irmse': irmse, "rel": rel}
