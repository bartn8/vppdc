# Data loading based on https://github.com/NVIDIA/flownet2-pytorch

import numpy as np
import torch
import torch.utils.data as data

import random
from glob import glob
import os.path as osp
import h5py

import frame_utils

from filter import filter_heuristic_depth
   
class KITTIDCVALDataset(data.Dataset):
    def __init__(self, datapath, test=False, overfit=False, baseline=None, topcrop = 100):
        self.topcrop = topcrop

        self.is_test = test
        self.init_seed = False
        self.image_list = []
        self.extra_info = []
        self.calib_list = []
        self.baseline = baseline

        # Glue code between persefone data and my shitty format
        image2_list = sorted(glob(osp.join(datapath, 'image/*.png')))
        image3_list = sorted(glob(osp.join(datapath, 'image/*.png')))
        gt_list = sorted(glob(osp.join(datapath, 'groundtruth_depth/*.png')))
        hints_list = sorted(glob(osp.join(datapath, 'velodyne_raw/*.png')))
        calibtxt_list = sorted(glob(osp.join(datapath, 'intrinsics/*.txt')))

        for i in range(len(image2_list)):
                    self.image_list += [ [image2_list[i], image3_list[i], gt_list[i], hints_list[i]] ]
                    self.extra_info += [ image2_list[i].split('/')[-1] ] # scene and frame_id
                    self.calib_list += [ calibtxt_list[i] ]

        if overfit:
            self.image_list = self.image_list[0:1]
            self.extra_info = self.extra_info[0:1]

    def __getitem__(self, index):

        if not self.init_seed and not self.is_test:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        data = {}

        data['im2'] = frame_utils.read_gen(self.image_list[index][0])
        data['im3'] = frame_utils.read_gen(self.image_list[index][1])
        
        data['im2'] = np.array(data['im2']).astype(np.uint8)
        data['im3'] = np.array(data['im3']).astype(np.uint8)

        data['im2'] = data['im2'][self.topcrop:,:,:]
        data['im3'] = data['im3'][self.topcrop:,:,:]

        if self.is_test:
            data['im2'] = data['im2'] / 255.0
            data['im3'] = data['im3'] / 255.0

        # grayscale images
        if len(data['im2'].shape) == 2:
            data['im2'] = np.tile(data['im2'][...,None], (1, 1, 3))
        else:
            data['im2'] = data['im2'][..., :3]                

        if len(data['im3'].shape) == 2:
            data['im3'] = np.tile(data['im3'][...,None], (1, 1, 3))
        else:
            data['im3'] = data['im3'][..., :3]

        #Read depth 
        gt_depth, data['validgt'] = frame_utils.readDispKITTI(self.image_list[index][2]) 
        hints_depth, data['validhints'] = frame_utils.readDispKITTI(self.image_list[index][3])    

        gt_depth = gt_depth[self.topcrop:,:,:]
        data['validgt'] = data['validgt'][self.topcrop:,:,:]
        hints_depth = hints_depth[self.topcrop:,:,:]
        data['validhints'] = data['validhints'][self.topcrop:,:,:]

        f_02 = -1

        f = open(self.calib_list[index], "r")
        line = f.readlines()[0]
        f.close()

        f_02 = float(line.strip().split(" ")[0])
            
        baseline = 0.53275
        baseline = baseline if self.baseline is None else self.baseline

        fhints_depth, _ = np.expand_dims(filter_heuristic_depth(hints_depth[..., 0], nx=7, ny=7, th=1.5), -1)
        fvalidhints_depth = np.where(fhints_depth > 0, 1, 0)
        data['fvalidhints'] = np.logical_and(fvalidhints_depth, data['validhints']).astype(np.uint8)

        #Convert depth to disparity
        data['gt'] = gt_depth.copy()
        data['hints'] = hints_depth.copy()
        data['fhints'] = fhints_depth.copy()
        
        data['gt_depth'] = gt_depth
        data['gt'][data['validgt'] > 0] = (f_02*baseline) / data['gt'][data['validgt'] > 0]
        data['hints'][data['validhints'] > 0] = (f_02*baseline) / hints_depth[data['validhints'] > 0]
        data['fhints'][data['fvalidhints'] > 0] = (f_02*baseline) / fhints_depth[data['fvalidhints'] > 0]
        
        if self.is_test:
            data['im2_aug'] = data['im2']
            data['im3_aug'] = data['im3']
        else:
            raise NotImplementedError
                    
        for k in data:
            if data[k] is not None:
                assert len(data[k].shape) == 3, f"k:{k}, shape: {data[k].shape}"
                data[k] = torch.from_numpy(data[k]).permute(2, 0, 1).float() 

        data['extra_info'] = self.extra_info[index]
        data['calib_data'] = {'f': f_02, 'b': baseline}

        return data

    def __len__(self):
        return len(self.image_list)

class NYUDepthVALV2Dataset(data.Dataset):
    def __init__(self, datapath, test=False, overfit=False, baseline=None):
        self.baseline = baseline

        self.is_test = test
        self.init_seed = False
        self.image_list = []
        self.extra_info = []

        # Glue code between persefone data and my shitty format
        h5_list = sorted(glob(osp.join(datapath, '*/*.h5')))

        for i in range(len(h5_list)):
                    self.image_list += [ [h5_list[i]] ]
                    self.extra_info += [ h5_list[i].split('/')[-1] ] # scene and frame_id

        if overfit:
            self.image_list = self.image_list[0:1]
            self.extra_info = self.extra_info[0:1]

    def __getitem__(self, index):

        if not self.init_seed and not self.is_test:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        data = {}

        hf = h5py.File("/mnt/massa1/datasets/nyudepthv2/nyu_img_gt.h5", 'r')
        gt_depth = hf['gt'][index]
        rgb = hf['img'][index]
        hf.close()

        hf = h5py.File("/mnt/massa1/datasets/nyudepthv2/nyu_pred_with_500.h5", 'r')
        hints_depth = hf['hints'][index]
        hf.close()

        data['im2'] = rgb.astype(np.uint8)
        data['im3'] = rgb.astype(np.uint8)

        if self.is_test:
            data['im2'] = data['im2'] / 255.0
            data['im3'] = data['im3'] / 255.0
        else:
            raise NotImplementedError     

        f_02 = 518.85790117450188 / 2
        #baseline = 0.075 # Kinect baseline
        baseline = 0.15
        baseline = baseline if self.baseline is None else self.baseline

        #Convert depth to disparity
        data['gt'] = gt_depth.copy()
        data['validgt'] = np.where(gt_depth > 0.0,1,0).astype(np.uint8)

        data['gt_depth'] = gt_depth
        data['gt'][data['validgt'] > 0] = (f_02*baseline) / data['gt'][data['validgt'] > 0]

        data['hints'] = hints_depth.copy()
        data['validhints'] = np.where(hints_depth > 0.0,1,0).astype(np.uint8)
        data['hints'][data['validhints'] > 0] = (f_02*baseline) / hints_depth[data['validhints'] > 0]
        
        if self.is_test:
            data['im2_aug'] = data['im2']
            data['im3_aug'] = data['im3']
        else:
            raise NotImplementedError

        for k in data:
            if data[k] is not None:
                assert len(data[k].shape) == 3, f"k:{k}, shape: {data[k].shape}"
                data[k] = torch.from_numpy(data[k]).permute(2, 0, 1).float() 
        
        data['extra_info'] = self.extra_info[index]

        data['calib_data'] = {'f': f_02, 'b': baseline}

        return data

    def __len__(self):
        return len(self.image_list)

class VOIDDataset(data.Dataset):
    def __init__(self, datapath, test=False, overfit=False, baseline=None):
        self.baseline = baseline

        self.is_test = test
        self.init_seed = False
        self.image_list = []
        self.extra_info = []

        appendix = "test" if test else "train"

        # Glue code between persefone data and my shitty format
        intrinsics_txt = open(osp.join(datapath, f"{appendix}_intrinsics.txt"), 'r')
        rgb_txt = open(osp.join(datapath, f"{appendix}_image.txt"), 'r')
        hints_txt = open(osp.join(datapath, f"{appendix}_sparse_depth.txt"), 'r')
        gt_txt = open(osp.join(datapath, f"{appendix}_ground_truth.txt"), 'r')
        valid_txt = open(osp.join(datapath, f"{appendix}_validity_map.txt"))

        while True:

            i_path = intrinsics_txt.readline().strip()
            rgb_path = rgb_txt.readline().strip()
            hints_path = hints_txt.readline().strip()
            gt_path = gt_txt.readline().strip()
            valid_path = valid_txt.readline().strip()

            if not i_path or not rgb_path or not hints_path or not gt_path or not valid_path:
                break

            self.image_list += [ [osp.join(datapath, i_path),
                                  osp.join(datapath, rgb_path),
                                   osp.join(datapath, hints_path),
                                     osp.join(datapath, gt_path),    
                                       osp.join(datapath, valid_path)] ]
            self.extra_info += [ [rgb_path.split('/')[-1]] ]

        intrinsics_txt.close()
        rgb_txt.close()
        hints_txt.close()
        gt_txt.close()
        valid_txt.close()

        if overfit:
            self.image_list = self.image_list[0:1]
            self.extra_info = self.extra_info[0:1]

    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, index):
        if not self.init_seed and not self.is_test:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        data = {}

        K = np.loadtxt(self.image_list[index][0])
        rgb = frame_utils.read_gen(self.image_list[index][1])
        hints_depth, _ = frame_utils.readDispKITTI(self.image_list[index][2]) 
        gt_depth, _ = frame_utils.readDispKITTI(self.image_list[index][3]) 

        data['im2'] = np.array(rgb).astype(np.uint8)
        data['im3'] = np.array(rgb).astype(np.uint8)

        data['im2'] = data['im2'].astype(np.uint8)
        data['im3'] = data['im3'].astype(np.uint8)

        if self.is_test:
            data['im2'] = data['im2'] / 255.0
            data['im3'] = data['im3'] / 255.0

        f_02 = K[0,0]
        #baseline = 0.05 # Intel Realsense D435i
        baseline = 0.1
        baseline = baseline if self.baseline is None else self.baseline

        #Convert depth to disparity
        data['gt'] = gt_depth.copy()
        data['validgt'] = np.where(gt_depth > 0.0,1.0,0).astype(np.uint8)
        data['gt_depth'] = gt_depth
        data['gt'][data['validgt'] > 0] = (f_02*baseline) / data['gt'][data['validgt'] > 0]

        data['hints'] = hints_depth.copy()
        data['validhints'] = np.where(hints_depth > 0.0,1.0,0).astype(np.uint8)
        data['hints'][data['validhints'] > 0] = (f_02*baseline) / hints_depth[data['validhints'] > 0]
        
        if self.is_test:
            data['im2_aug'] = data['im2']
            data['im3_aug'] = data['im3']
        else:
            raise NotImplementedError

        for k in data:
            if data[k] is not None:
                assert len(data[k].shape) == 3, f"k:{k}, shape: {data[k].shape}"
                data[k] = torch.from_numpy(data[k]).permute(2, 0, 1).float() 
        
        data['extra_info'] = self.extra_info[index]

        data['calib_data'] = {'f': f_02, 'b': baseline}

        return data   
    
class MYDDADDataset(data.Dataset):
    def __init__(self, datapath, test=False, overfit=False, baseline=None, topcrop = 0, bottomcrop = 0):
        self.topcrop = topcrop
        self.bottomcrop = bottomcrop
        self.baseline = baseline

        self.is_test = test
        self.init_seed = False

        self.image_list = []
        self.extra_info = []
        self.calib_list = []

        # Glue code between persefone data and my shitty format
        image_list = sorted(glob(osp.join(datapath, 'rgb/*.png')))
        gt_list = sorted(glob(osp.join(datapath, '*gt/*.png')))
        hints_list = sorted(glob(osp.join(datapath, 'hints/*.png')))
        calibtxt_list = sorted(glob(osp.join(datapath, 'intrinsics/*.txt')))

        #Filter data not present in other folders
        for i in range(len(image_list)):
            self.image_list += [ [image_list[i], image_list[i], gt_list[i], hints_list[i]] ]
            self.extra_info += [ [image_list[i].split('/')[-1], False] ] # scene and frame_id and do flip
            self.calib_list += [ [calibtxt_list[i]] ]

        if overfit:
            self.image_list = self.image_list[0:1]
            self.extra_info = self.extra_info[0:1]

    def __getitem__(self, index):

        if not self.init_seed and not self.is_test:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        data = {}

        K = np.loadtxt(self.calib_list[index][0])
        f_02 = K[0,0]
        baseline = 0.3
        baseline = baseline if self.baseline is None else self.baseline

        data['im2'] = frame_utils.read_gen(self.image_list[index][0])
        data['im3'] = frame_utils.read_gen(self.image_list[index][1])
        
        data['im2'] = np.array(data['im2']).astype(np.uint8)
        data['im3'] = np.array(data['im3']).astype(np.uint8)

        if self.is_test:
            data['im2'] = data['im2'] / 255.0
            data['im3'] = data['im3'] / 255.0

        # grayscale images
        if len(data['im2'].shape) == 2:
            data['im2'] = np.tile(data['im2'][...,None], (1, 1, 3))
        else:
            data['im2'] = data['im2'][..., :3]                

        if len(data['im3'].shape) == 2:
            data['im3'] = np.tile(data['im3'][...,None], (1, 1, 3))
        else:
            data['im3'] = data['im3'][..., :3]

        h,_ = data['im2'].shape[:2]
        data['im2'] = data['im2'][self.topcrop:h-self.bottomcrop,:,:]
        data['im3'] = data['im3'][self.topcrop:h-self.bottomcrop,:,:]

        #Read depth 
        gt_depth, data['validgt'] = frame_utils.readDispKITTI(self.image_list[index][2]) 
        hints_depth, data['validhints'] = frame_utils.readDispKITTI(self.image_list[index][3])         

        gt_depth = gt_depth[self.topcrop:h-self.bottomcrop,:,:]
        data['validgt'] = data['validgt'][self.topcrop:h-self.bottomcrop,:,:]
        hints_depth = hints_depth[self.topcrop:h-self.bottomcrop,:,:]
        data['validhints'] = data['validhints'][self.topcrop:h-self.bottomcrop,:,:]

        fhints_depth, _ = np.expand_dims(filter_heuristic_depth(hints_depth[..., 0], nx=7, ny=7, th=1.5), -1)
        fvalidhints_depth = np.where(fhints_depth > 0, 1, 0)
        data['fvalidhints'] = np.logical_and(fvalidhints_depth, data['validhints']).astype(np.uint8)

        #Convert depth to disparity
        data['gt'] = gt_depth.copy()
        data['hints'] = hints_depth.copy()
        data['fhints'] = fhints_depth.copy()

        data['gt_depth'] = gt_depth
        data['gt'][data['validgt'] > 0] = (f_02*baseline) / data['gt'][data['validgt'] > 0]
        data['hints'][data['validhints'] > 0] = (f_02*baseline) / hints_depth[data['validhints'] > 0]
        data['fhints'][data['fvalidhints'] > 0] = (f_02*baseline) / fhints_depth[data['fvalidhints'] > 0]
        
        if self.is_test:
            data['im2_aug'] = data['im2']
            data['im3_aug'] = data['im3']
        else:
            raise NotImplementedError
            
        for k in data:
            if data[k] is not None:
                assert len(data[k].shape) == 3, f"k:{k}, shape: {data[k].shape}"
                data[k] = torch.from_numpy(data[k]).permute(2, 0, 1).float() 
        
        data['extra_info'] = self.extra_info[index]
        data['calib_data'] = {'f': f_02, 'b': baseline}

        return data

    def __len__(self):
        return len(self.image_list)

#----------------------------------------------------------------------------------------------

def worker_init_fn(worker_id):                                                          
    torch_seed = torch.initial_seed()
    random.seed(torch_seed + worker_id)
    if torch_seed >= 2**30:  # make sure torch_seed + workder_id < 2**32
        torch_seed = torch_seed % 2**30
    np.random.seed(torch_seed + worker_id)

def fetch_dataloader(args):
    """ Create the data loader for the corresponding trainign set """

    if args.dataset == 'kittidcval':
        if args.test:
            dataset = KITTIDCVALDataset(args.datapath,test=True,baseline=args.baseline)
            loader = data.DataLoader(dataset, batch_size=args.batch_size, 
                pin_memory=False, shuffle=False, num_workers=8, drop_last=True)
            print(f'Testing using {args.dataset} with {len(loader.dataset)} image pairs')
        else:
            raise NotImplementedError   

    elif args.dataset == 'nyudepthv2':
        if args.test:
            dataset = NYUDepthVALV2Dataset(args.datapath,test=True,baseline=args.baseline)
            loader = data.DataLoader(dataset, batch_size=args.batch_size, 
                pin_memory=False, shuffle=False, num_workers=8, drop_last=True)
            print(f'Testing using {args.dataset} with {len(loader.dataset)} image pairs')
        else:
            raise NotImplementedError
        
    elif args.dataset == 'void':
        if args.test:
            dataset = VOIDDataset(args.datapath,test=True,baseline=args.baseline)
            loader = data.DataLoader(dataset, batch_size=args.batch_size, 
                pin_memory=False, shuffle=False, num_workers=8, drop_last=True)
            print(f'Testing using {args.dataset} with {len(loader.dataset)} image pairs')
        else:
            raise NotImplementedError

    elif args.dataset == 'myddad':
        if args.test:
            dataset = MYDDADDataset(args.datapath,test=True,baseline=args.baseline)
            loader = data.DataLoader(dataset, batch_size=args.batch_size, 
                pin_memory=False, shuffle=False, num_workers=8, drop_last=True)
            print(f'Testing using {args.dataset} with {len(loader.dataset)} image pairs')
        else:
            raise NotImplementedError
         
    return loader
