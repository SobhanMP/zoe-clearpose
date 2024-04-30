# MIT License

import glob
import itertools
from pathlib import Path
import pathlib
import sys
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os

from PIL import Image
import numpy as np
import cv2


clearpose_sets = {
    "train": ['set1','set4','set5','set6','set7'],
    "test": ['set2', 'set3', 'set8', 'set9']
} 

class ToTensor(object):
    def __init__(self, dataset):
        # self.normalize = transforms.Normalize(
        #     mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.normalize = lambda x: x
        self.dataset = dataset
        # self.resize = transforms.Resize((375, 1242))

    def __call__(self, sample):
        image, camera_depth, depth = sample['image'], sample['camera_depth'], sample['depth']

        image = self.to_tensor(image)
        image = self.normalize(image)
        depth = self.to_tensor(depth)
        camera_depth = self.to_tensor(camera_depth)
        # image = self.resize(image)

        return {'image': image, 'depth': depth, 'camera_depth': camera_depth, 'dataset': self.dataset, 'mask': depth != 0}

    def to_tensor(self, pic):
        if isinstance(pic, np.ndarray):
            # print('np.ndarray', pic.dtype)
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            return img

        #         # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(
                torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)

        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            return img.float()
        else:
            return img


class ClearPose(Dataset):
    def __init__(self, data_dir_root, do_kb_crop=True, split="test"):
        self.fns = ['-'.join(str(i).split('-')[:-1]) 
                    for i in itertools.chain.from_iterable((Path(data_dir_root) / i).rglob("*-label.png") for i in clearpose_sets[split])]

        self.transform = ToTensor("clearpose")

    def __getitem__(self, idx):

        fn = self.fns[idx]

        image = Image.open(fn + "-color.png")
        depth = cv2.imread(fn + "-depth_true.png",  cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        camera_depth = cv2.imread(fn + "-depth.png",  cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        
        # print("depth min max", depth.min(), depth.max())


        # print(np.shape(image))
        # print(np.shape(depth))

        # depth[depth > 8] = -1

      
        image = np.asarray(image, dtype=np.float32) / 255.0
        # print( image.min(), image.max(), image.dtype)
        depth = depth[..., None].astype(np.float32) / 1000.0 
        camera_depth = camera_depth[..., None].astype(np.float32) / 1000.0
        sample = dict(image=image, depth=depth, camera_depth=camera_depth)

        # return sample
        sample = self.transform(sample)

        if idx == 0:
            print('image shape is:', sample["image"].shape)

        return sample

    def __len__(self):
        return len(self.fns)


def get_clearpose_loader(data_dir_root, batch_size=1, **kwargs):
    dataset = ClearPose(data_dir_root)
    return DataLoader(dataset, batch_size, **kwargs)


if __name__ == "__main__":
    dataset_dir = sys.argv[1]
    
    loader = get_clearpose_loader(dataset_dir)
    print('log', "Total files", len(loader.dataset))
    for i, sample in enumerate(loader):
        print('log', sample["image"].shape)
        print('log', sample["depth"].shape)
        print('log', sample["dataset"])
        print('log', sample['depth'].min(), sample['depth'].max())
        if i > 5:
            break
