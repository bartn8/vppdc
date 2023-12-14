from __future__ import print_function
import matplotlib.pyplot as plt
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from visualizer import color_error_image_kitti, add_text_and_rectangle

from models.raft_stereo import *

from torchvision.transforms.functional import rgb_to_grayscale


from vpp_standalone import vpp
from filter import occlusion_heuristic
from utils import sgm_opencv

from losses import *
import cv2
import sys

sys.path.append('dataloaders')
import datasets
import tqdm

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

parser = argparse.ArgumentParser(description='VPP for DC')

parser.add_argument('--maxdisp', type=int ,default=256,
                    help='maxium disparity')

parser.add_argument('--model', default='raft-stereo',
                    help='select model')

parser.add_argument('--loadmodel',
                    help='load model', default=None)

parser.add_argument('--datapath', default='dataset/oak_dataset/',
                    help='datapath')

parser.add_argument('--dataset', default='middlebury', help='dataset type')

parser.add_argument('--outdir', default=None)  
        
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')

parser.add_argument('--iscale', type=int, default=1, 
                    help='Downsampling factor')
parser.add_argument('--oscale', type=int, default=1,
                            help='Downsampling factor')

parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--wsize', type=int, default=5, help='Patch size')
parser.add_argument('--guideperc', type=float, default=0.05)
parser.add_argument('--blending', type=float, default=1, help='Pattern alpha blending')
parser.add_argument('--valsize', type=int, default=0, help='validation max size (0=unlimited)')
parser.add_argument('--normalize', action='store_true')
parser.add_argument('--maskocc', action='store_true')
parser.add_argument('--cblending', type=float, default=0.0, help='Pattern alpha blending on occluded points')
parser.add_argument('--uniform_color', action='store_true')
parser.add_argument('--guided', action='store_true')
parser.add_argument('--tries', type=int, default=1)

parser.add_argument('--filterlidar', action='store_true')
parser.add_argument('--z_max', type=float, default=0.0)  
parser.add_argument('--z_clip', type=float, default=0.0)  
parser.add_argument('--refdomain', choices=['depth', 'disparity'], default='depth')
parser.add_argument('--refbins', type=int, default=128)
parser.add_argument('--gt_source', action='store_true')
parser.add_argument('--interpolate', action='store_true')
parser.add_argument('--filling', action='store_true')
parser.add_argument('--leftpadding', action='store_true')
parser.add_argument('--context', action='store_true')
parser.add_argument('--baseline', type=float, default=None)  

parser.add_argument('--o_xy', type=float, default=1)  
parser.add_argument('--o_i', type=float, default=1)  
parser.add_argument('--th_adpt', type=float, default=0.001) 

parser.add_argument('--depth_only', action='store_true')

parser.add_argument('--hints_dilation', type=int, default=0)
parser.add_argument('--errormap_dilation', type=int, default=0)
parser.add_argument('--errormap_scale', type=float, default=1)

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)

if args.cuda:
    torch.cuda.manual_seed(args.seed)
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

stereonet = None

if args.model == 'raft-stereo':
    stereonet = RAFTStereo(args)
elif args.model in ['sgm']:
    stereonet = None
else:
    print(f'no model ({args.model}:{args.loadmodel})')
    exit()

if args.cuda:
    if stereonet is not None:  
        stereonet = nn.DataParallel(stereonet)
        stereonet.to(device)

if args.loadmodel is not None:
    print('Load pretrained stereo model...')

    if args.model in ['raft-stereo']:
        state_dict = torch.load(args.loadmodel, map_location=device)
        state_dict  = state_dict['state_dict'] if 'state_dict' in state_dict else state_dict
        stereonet.load_state_dict(state_dict)
    else:
        print(f"Cannot load {args.model}")
elif stereonet is not None:
    print('No pretrained stereo model')
    exit()

n_params_stereo = sum([p.data.nelement() for p in stereonet.parameters()]) if stereonet is not None else 0
print('Number of stereo model parameters: {}'.format(n_params_stereo))

@torch.no_grad()
def run(data):
    global stereonet

    if args.model not in ['sgm']:
        stereonet.eval()
    
    if 'hints' not in data or args.gt_source:   
        data['hints'], data['validhints'] = sample_hints(data['gt'], data['validgt']>0, probability=args.guideperc) 
    else:
        data['hints'], data['validhints'] = sample_hints(data['hints'], data['validhints']>0, probability=args.guideperc) 

    if 'fhints' not in data or not args.filterlidar:
        data['fhints'], data['fvalidhints'] = data['hints'], data['validhints']

    baseline,focal = data['calib_data']['b'][0].item(), data['calib_data']['f'][0].item()

    run.counter +=1

    if args.iscale != 1:
        for k in ['im2', 'im3']:
            data[k] = F.interpolate(data[k], scale_factor=1./args.iscale)
        
        for k in ['hints', 'fhints']:
            data[k] = F.interpolate(data[k], scale_factor=1./args.iscale, mode='nearest') / args.iscale

        tmp_map = {'validhints':'hints','fvalidhints':'fhints'}

        for k in ['validhints', 'fvalidhints']:
            data[k] = torch.where(data[tmp_map[k]] > 0,1,0)

    if args.oscale != 1:
        data['gt'] = F.interpolate(data['gt'], scale_factor=1./args.oscale, mode='nearest-exact') / args.oscale
        data['validgt'] = F.interpolate(data['validgt'], scale_factor=1./args.oscale, mode='nearest-exact')
            
    #Prepadding to add left border occlusion points
    w = data['hints'].shape[-1]
    left_pad_size = round(data['hints'].max().item())
    left_pad_size = min(left_pad_size, 200)        
    left_pad_size = left_pad_size + (32-(left_pad_size+w) % 32)

    prepad = args.leftpadding
    left_pad_size = left_pad_size if prepad else 0
    _prepad = [left_pad_size,0,0,0]
        
    for k in ['im2', 'im3', 'hints', 'validhints', 'fhints', 'fvalidhints']:
        data[k] = F.pad(data[k], _prepad, mode='constant', value=0)
    
    c,h,w = data['im2'][0].shape
    left_black = np.zeros((h,w,c), dtype=np.uint8)
    right_black = np.zeros((h,w,c), dtype=np.uint8)

    im2_blended_list = []
    im3_blended_list = []

    for b in range(args.batch_size):
        left = (255*data['im2'][b].permute(1,2,0).numpy()).astype(np.uint8)
        extrapolated_hints = data['fhints'][b,0].numpy()
        mask_occ = occlusion_heuristic(extrapolated_hints)[1] if args.maskocc else None

        im2_blended, im3_blended = vpp(left, left_black, right_black,
                                                extrapolated_hints, blending=args.blending, wsize=args.wsize,
                                                c_occ=args.cblending, g_occ=mask_occ, useFilling = args.filling, useContext=args.context,
                                                fillingThreshold=args.th_adpt, o_xy=args.o_xy, o_i=args.o_i,
                                                left2right=True, method='rnd', uniform_color=args.uniform_color, interpolate=args.interpolate )
    
        im2_blended_list.append(torch.from_numpy(im2_blended/255.).permute(2,0,1).unsqueeze(0).float())
        im3_blended_list.append(torch.from_numpy(im3_blended/255.).permute(2,0,1).unsqueeze(0).float())
        
    data['im2_blended'] = torch.cat(im2_blended_list,0)
    data['im3_blended'] = torch.cat(im3_blended_list,0)

    if args.cuda:
        for k in ['im2', 'im3', 'im2_blended', 'im3_blended', 'hints', 'validhints', 'fhints', 'fvalidhints']:
            data[k] = data[k].cuda()
    
    ht, wt = data['im2'].shape[-2], data['im2'].shape[-1]

    if args.model in ['raft-stereo', 'sgm']:
        pad_ht = (((ht // 32) + 1) * 32 - ht) % 32
        pad_wd = (((wt // 32) + 1) * 32 - wt) % 32
    
    #Stereo network prepadding
    _pad = [pad_wd//2, pad_wd - pad_wd//2, pad_ht//2, pad_ht - pad_ht//2]
    
    for k in ['im2', 'im3', 'im2_blended', 'im3_blended', 'hints', 'validhints', 'fhints', 'fvalidhints']:
        data[k] = F.pad(data[k], _pad, mode='constant', value=0)

    data['im2_vpp'], data['im3_vpp'] = data['im2_blended'] , data['im3_blended']

    inject_hints = data['fhints'] if args.guided else None
    inject_validhints = data['fvalidhints'] if args.guided else None

    if args.model == 'sgm':
        pred_disp = []
        left_gray = rgb_to_grayscale(data['im2_vpp'])
        right_gray = rgb_to_grayscale(data['im3_vpp'])

        for b in range(data['im2'].shape[0]):
            left = (left_gray[b,0].cpu().numpy()*255.).astype(np.uint8)
            right = (right_gray[b,0].cpu().numpy()*255.).astype(np.uint8)
            disp = sgm_opencv(left,right,args.maxdisp)
            pred_disp.append(torch.from_numpy(disp).unsqueeze(0).unsqueeze(0).float())
        
        pred_disps = torch.cat(pred_disp, 0).to(device)
        
    elif args.model in ['raft-stereo']:
        ctx = data['im2'] if not args.depth_only else torch.zeros_like(data['im2'])
        _,pred_disps = stereonet(ctx, data['im2_vpp'], data['im3_vpp'], 
        test_mode=True, iters=22, normalize=args.normalize,
        hints=inject_hints, validhints=inject_validhints)
    else:
        pred_disps = stereonet(data['im2_vpp'], data['im3_vpp'])
    
    if args.model == 'raft-stereo':
        pred_disp = -pred_disps.squeeze(1)
    elif args.model in ['sgm']:
        pred_disp = pred_disps.squeeze(1)

    ht, wd = pred_disp.shape[-2:]
    c = [_pad[2], ht-_pad[3], _pad[0], wd-_pad[1]]
    pred_disp = pred_disp[..., c[0]:c[1], c[2]:c[3]]
    
    for k in ['fhints', 'fvalidhints', 'hints', 'validhints', 'im2', 'im3']:
        data[k] = data[k][..., c[0]:c[1], c[2]:c[3]]

    #Remove left border prepadding
    ht, wd = pred_disp.shape[-2:]
    c = [_prepad[2], ht-_prepad[3], _prepad[0], wd-_prepad[1]]
    pred_disp = pred_disp[..., c[0]:c[1], c[2]:c[3]]

    for k in ['fhints', 'fvalidhints', 'hints', 'validhints', 'im2', 'im3']:
        data[k] = data[k][..., c[0]:c[1], c[2]:c[3]]

    if args.iscale != 1:
        for k in ['im2', 'im3']:
            data[k] = F.interpolate(data[k], scale_factor=args.iscale)
        
        for k in ['hints', 'fhints']:
            data[k] = F.interpolate(data[k], scale_factor=args.iscale, mode='nearest') * args.iscale
        
        for k in ['validhints', 'fvalidhints']:
            data[k] = F.interpolate(data[k].float(), scale_factor=args.iscale, mode='nearest')

    if args.iscale != 1 and args.iscale/args.oscale != 1:
        pred_disp = F.interpolate(pred_disp.unsqueeze(0), scale_factor=args.iscale/args.oscale, mode='nearest').squeeze(0) * args.iscale / args.oscale

    pred_depth = pred_disp.clone()
    pred_depth[pred_depth>0] = (focal*baseline) / pred_depth[pred_depth>0]
    pred_depth[pred_depth<=0] = 0

    if args.z_clip > 0:
        pred_depth[pred_depth>args.z_clip] = args.z_clip

    if args.refdomain == 'depth':
        if 'gt_depth' in data:
            gt = data['gt_depth']
        else:
            gt = data['gt'].clone()
            gt[gt>0] = (focal*baseline) / gt[gt>0]
        gt[gt<=0] = 0
        hints = data['hints'].clone()
        hints[hints>0] = (focal*baseline) / hints[hints>0]
        hints[hints<=0] = 0
        fhints = data['fhints'].clone()
        fhints[fhints>0] = (focal*baseline) / fhints[fhints>0]
        fhints[fhints<=0] = 0
        pred = pred_depth
    elif args.refdomain == 'disparity':
        gt = data['gt']
        hints = data['hints']
        fhints = data['fhints']
        pred = pred_disp
    else:
        print(f"error refdomain: {args.refdomain}")

    hints = hints.squeeze().unsqueeze(0)
    fhints = fhints.squeeze().unsqueeze(0)             

    if args.refdomain == 'depth':
        result = kitti_depth_metrics(pred.cpu().numpy(), gt.numpy(), (gt>0).numpy(), max_depth = args.z_max, clip_depth=args.z_clip)
    else:
        result = kitti_depth_metrics(pred.cpu().numpy(), gt.numpy(), (gt>0).numpy(), baseline, focal, args.z_max, clip_depth=args.z_clip)
        
    result['pred'] = pred.squeeze()
    result['hints'] = hints.squeeze()
    result['fhints'] = fhints.squeeze()
    result['gt'] = gt.squeeze()
    result['im2_vpp'] = 255*data['im2_vpp'].squeeze().permute(1,2,0)
    result['im3_vpp'] = 255*data['im3_vpp'].squeeze().permute(1,2,0)

    return result

def main():
    args.test = True
    args.batch_size = 1
    demo_loader = datasets.fetch_dataloader(args)
    
    if args.outdir is not None and not os.path.exists(args.outdir):
        os.mkdir(args.outdir)

    ## demo ##
    acc_list = []
    run.counter = 0
    
    for asd in range(args.tries):

        acc = {}
        pbar = tqdm.tqdm(total=len(demo_loader))
        val_len = min(len(demo_loader), args.valsize) if args.valsize > 0 else len(demo_loader)
        for batch_idx, datablob in enumerate(demo_loader):
            if batch_idx >= val_len:
                break

            result = run(datablob)

            if args.outdir is not None and asd == 0:
                for dirname in ['pred', 'left', 'right', 'errormap', 'errormap_text', 'hints', 'fhints', 'context', 'gt']:
                    if not os.path.exists(os.path.join(args.outdir, dirname)):
                        os.mkdir(os.path.join(args.outdir, dirname))

                errormap = torch.abs(result['pred'].cpu()-result['gt'])
                errormap[result['gt'] <= 0] = 0
                # errormap[datablob['hints'][0,0] <= 0] = 0
                errormap = errormap.detach().cpu().numpy()

                max_depth = args.z_max
                max_depth = max_depth if max_depth > 0 else None

                if max_depth is not None:
                    maxval = min(torch.max(result['gt']).item(), max_depth)
                else:
                    maxval = torch.max(result['gt']).item()

                minval = torch.min(result['gt']).item()
                

                pred = torch.clamp(result['pred'], minval, max_depth) / maxval
                pred = (maxval * pred).cpu().numpy()

                hints = torch.clamp(result['hints'], minval, max_depth) / maxval
                hints = (maxval * hints).cpu().numpy()

                fhints = torch.clamp(result['fhints'], minval, max_depth) / maxval
                fhints = (maxval * fhints).cpu().numpy()

                gt = torch.clamp(result['gt'], 0, max_depth) / maxval
                gt = (maxval * gt).cpu().numpy()

                if args.hints_dilation > 0:
                    kernel = np.ones((args.hints_dilation, args.hints_dilation))
                    hints = cv2.dilate(hints, kernel)
                    fhints = cv2.dilate(fhints, kernel)

                if args.errormap_dilation > 0: 
                    kernel = np.ones((args.errormap_dilation, args.errormap_dilation))
                    gt = cv2.dilate(gt, kernel)

                errormap_img = cv2.applyColorMap((255*errormap/np.max(errormap)).astype(np.uint8), cv2.COLORMAP_MAGMA)
                errormap_img = (errormap_img * 0.7 + 0.3 * cv2.cvtColor((255*datablob['im2'].squeeze().permute(1,2,0).detach().cpu().numpy()).astype(np.uint8), cv2.COLOR_RGB2BGR)).astype(np.uint8)
                errormap_img = cv2.putText(errormap_img, f"MAE: {result['mae']:.3f}", (5,15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
                errormap_img = cv2.putText(errormap_img, f"RMSE: {result['rmse']:.3f}", (5,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
                
                plt.imsave(os.path.join(os.path.join(args.outdir, "pred"), '%s.png'%(batch_idx)), pred, cmap='magma', vmin=minval, vmax=maxval)
                plt.imsave(os.path.join(os.path.join(args.outdir, "hints"), '%s.png'%(batch_idx)), hints, cmap='magma', vmin=minval, vmax=maxval)
                plt.imsave(os.path.join(os.path.join(args.outdir, "fhints"), '%s.png'%(batch_idx)), fhints, cmap='magma', vmin=minval, vmax=maxval)
                plt.imsave(os.path.join(os.path.join(args.outdir, "gt"), '%s.png'%(batch_idx)), gt, cmap='magma', vmin=minval, vmax=maxval)

                cv2.imwrite(os.path.join(os.path.join(args.outdir, "context"), '%s.png'%(batch_idx)), cv2.cvtColor((255*datablob['im2'].squeeze().permute(1,2,0).detach().cpu().numpy()).astype(np.uint8), cv2.COLOR_RGB2BGR))
                cv2.imwrite(os.path.join(os.path.join(args.outdir, "left"), '%s.png'%(batch_idx)), cv2.cvtColor(result['im2_vpp'].squeeze().detach().cpu().numpy().astype(np.uint8), cv2.COLOR_RGB2BGR))
                cv2.imwrite(os.path.join(os.path.join(args.outdir, "right"), '%s.png'%(batch_idx)), cv2.cvtColor(result['im3_vpp'].squeeze().detach().cpu().numpy().astype(np.uint8), cv2.COLOR_RGB2BGR))

                errormap_img = color_error_image_kitti(errormap, args.errormap_scale, errormap>0, True, args.errormap_dilation)
                rect_width = gt.shape[1] // 5
                text_scale = 1 / 350 * rect_width
                errormap_img_text = add_text_and_rectangle(errormap_img.copy(), f"MAE: {result['mae']:.3f}m", rect_width, text_scale)
                cv2.imwrite(os.path.join(os.path.join(args.outdir, "errormap"), '%s.png'%(batch_idx)), errormap_img)
                cv2.imwrite(os.path.join(os.path.join(args.outdir, "errormap_text"), '%s.png'%(batch_idx)), errormap_img_text)


            for k in result:
                if k not in ['disp', 'errormap', 'errormap_text', 'im2_vpp', 'im3_vpp', 'hints', 'fhints', 'gt', 'pred']:
                    if k not in acc:
                        acc[k] = []
                    acc[k].append(result[k])

            pbar.update(1)
        pbar.close()

        acc_list.append(acc)
    
    acc_mean = {}
    acc_std = {}

    for acc in acc_list:
        for k in acc:
            if k not in acc_mean:
                acc_mean[k] = []
            if k not in acc_std:
                acc_std[k] = []
            
            acc_mean[k].append(np.array(acc[k]).mean())
            acc_std[k].append(np.array(acc[k]).mean())
    
    for k in acc_mean:
        acc_mean[k] = np.mean(acc_mean[k])
        acc_std[k] = np.std(acc_std[k])

    print("MEAN Metrics:")

    metrs = ''
    for k in acc_mean:
        metrs += f" {k.upper()} &"
    print(metrs)

    metrs = ''
    for k in acc_mean:
            if 'bad' not in k:
                metrs += f" {acc_mean[k]:.3f} &"
            else:
                metrs += f" {acc_mean[k]*100:.3f} &"

    print(metrs)

    print("STD Metrics:")

    metrs = ''
    for k in acc_std:
            if 'bad' not in k:
                metrs += f" {acc_std[k]:.3f} &"
            else:
                metrs += f" {acc_std[k]*100:.3f} &"

    print(metrs)


if __name__ == '__main__':
   main()
    
