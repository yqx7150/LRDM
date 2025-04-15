import glob
import logging
import os
import random

import albumentations as A
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import webdataset
from omegaconf import open_dict, OmegaConf
from skimage.feature import canny
from skimage.transform import rescale, resize
from torch.utils.data import Dataset, IterableDataset, DataLoader, DistributedSampler, ConcatDataset

from saicinpainting.evaluation.data import InpaintingDataset as InpaintingEvaluationDataset, \
    OurInpaintingDataset as OurInpaintingEvaluationDataset, ceil_modulo, InpaintingEvalOnlineDataset
from saicinpainting.training.data.aug import IAAAffine2, IAAPerspective2
from saicinpainting.training.data.masks import get_mask_generator

import os.path as osp
import scipy
import scipy.io as io
from scipy.io import loadmat , savemat
import sys

import mask_k


LOGGER = logging.getLogger(__name__)

def k2wgt(X,W):
    result = np.multiply(X,W)
    return result


def write_images(x,image_save_path):
    x = np.clip(x * 255, 0, 255).astype(np.uint8)
    cv2.imwrite(image_save_path, x)

class InpaintingTrainDataset(Dataset):
    def __init__(self, indir, mask_generator, transform):
        #self.in_files = list(glob.glob(os.path.join(indir, '**', '*.jpg'), recursive=True))
        self.in_files = list(glob.glob(os.path.join(indir, '**', '*.mat'), recursive=True))
        self.mask_generator = mask_generator
        self.transform = transform
        self.iter_i = 0

    def __len__(self):
        return len(self.in_files)

    def __getitem__(self, item):
        #print("start get one!!!!!!\n")
        path = self.in_files[item]
        #img = cv2.imread(path)
        #print("path = ",path)

        number=path.split('/')[-1].split('.')[0]

        #将数据读取出来
        siat_input =scipy.io.loadmat(path)
        siat_input = siat_input['Img2']   #256x256x3
        #print(type(siat_input),siat_input.shape)

        #转为k空间
        siat = np.array(siat_input[:,: , 0:2], dtype=np.float32)
        #print(siat.shape)
        siat_complex = siat[:, :, 0] + 1j * siat[:, :, 1]
        #print("siat_complex.shape=",siat_complex.shape)
        siat_kdata = np.fft.fft2(siat_complex)
        siat_kdata = np.fft.fftshift(siat_kdata)
        # data = {"kdata": siat_kdata}
        # savemat("/home/b109/code/xx/DiffIR-master/DiffIR-master/DiffIR-inpainting/data/temp/siat_kdata"+number+".mat" , data)


        #加载权重
        #weight = loadmat('/home/b109/Desktop/XX/inpainting/data/mri/weight/weight1_GEBrain.mat')['weight']
        weight = loadmat('/home/b109/Desktop/XX/inpainting/data/mri/weight/weight1_PeiBrain.mat')['weight']
        kdata_w = k2wgt(siat_kdata, weight)


        # data={"k_w":kdata_w}
        # savemat("/home/b109/code/xx/DiffIR-master/DiffIR-master/DiffIR-inpainting/data/temp/kw"+number+".mat" , data)

        #重新转为3x256x256
        siat_temp = np.zeros((256, 256, 3))
        kdata = np.array(siat_temp, dtype=np.float32)
        kdata[:, :, 0] = np.real(kdata_w)
        kdata[:, :, 1] = np.imag(kdata_w)
        kdata[:, :, 2] = np.real(kdata_w)
        kdata = kdata.transpose((2, 0, 1))      # 3x256x256

        #加载mask
        mask = np.zeros((1, 256, 256))
        #mask_item = io.loadmat('/home/b109/code/xx/DiffIR-master/DiffIR-master/DiffIR-inpainting/data/mask/cart/mask_cart_010.mat')['mask']
        #mask[0, :, :] = mask_item
        # mask = mask.astype(np.float32)
        #write_images(abs(mask_item), osp.join('/home/b109/code/xx/DiffIR-master/DiffIR-master/DiffIR-inpainting/data/temp/mask.png'))
        #mask = self.mask_generator(kdata, iter_i=self.iter_i)
        # print(mask)
        # sys.exit()

        #print("type(mask)=",mask[0][0][0].dtype,"      shape=",mask.shape)
        #print("type(mask)=",type(mask),"  shape=",mask.shape)

        #self.iter_i += 1

        mask[0, :, :] = mask_k.multiply_random_mask_k_space(kdata_w) 
        mask = 1-mask

        # mask_img = np.sqrt(np.sum(np.square(np.abs(mask)),axis=0))
        # mask_img = mask_img/np.max(np.abs(mask_img))
        # write_images(abs(mask_img),osp.join('/home/b109/code/xx/DiffIR-master/DiffIR-master/DiffIR-inpainting/data/temp/','mask'+'.png'))
        # sys.exit()
        #print(mask)
        mask = mask.astype(np.float32)





        # print("kdata = ",type(kdata)," ",kdata.shape,"  mask =",type(mask)," ",mask.shape)
        # exit(0)
        # sys.exit()
        return dict(image=kdata,
                    mask=mask)


class InpaintingTrainWebDataset(IterableDataset):
    def __init__(self, indir, mask_generator, transform, shuffle_buffer=200):
        self.impl = webdataset.Dataset(indir).shuffle(shuffle_buffer).decode('rgb').to_tuple('jpg')
        self.mask_generator = mask_generator
        self.transform = transform

    def __iter__(self):
        for iter_i, (img,) in enumerate(self.impl):
            img = np.clip(img * 255, 0, 255).astype('uint8')
            img = self.transform(image=img)['image']
            img = np.transpose(img, (2, 0, 1))
            mask = self.mask_generator(img, iter_i=iter_i)
            yield dict(image=img,
                       mask=mask)


class ImgSegmentationDataset(Dataset):
    def __init__(self, indir, mask_generator, transform, out_size, segm_indir, semantic_seg_n_classes):
        self.indir = indir
        self.segm_indir = segm_indir
        self.mask_generator = mask_generator
        self.transform = transform
        self.out_size = out_size
        self.semantic_seg_n_classes = semantic_seg_n_classes
        self.in_files = list(glob.glob(os.path.join(indir, '**', '*.jpg'), recursive=True))

    def __len__(self):
        return len(self.in_files)

    def __getitem__(self, item):
        path = self.in_files[item]
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.out_size, self.out_size))
        img = self.transform(image=img)['image']
        img = np.transpose(img, (2, 0, 1))
        mask = self.mask_generator(img)
        segm, segm_classes= self.load_semantic_segm(path)
        result = dict(image=img,
                      mask=mask,
                      segm=segm,
                      segm_classes=segm_classes)
        return result

    def load_semantic_segm(self, img_path):
        segm_path = img_path.replace(self.indir, self.segm_indir).replace(".jpg", ".png")
        mask = cv2.imread(segm_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (self.out_size, self.out_size))
        tensor = torch.from_numpy(np.clip(mask.astype(int)-1, 0, None))
        ohe = F.one_hot(tensor.long(), num_classes=self.semantic_seg_n_classes) # w x h x n_classes
        return ohe.permute(2, 0, 1).float(), tensor.unsqueeze(0)


def get_transforms(transform_variant, out_size):
    if transform_variant == 'default':
        transform = A.Compose([
            A.RandomScale(scale_limit=0.2),  # +/- 20%
            A.PadIfNeeded(min_height=out_size, min_width=out_size),
            A.RandomCrop(height=out_size, width=out_size),
            A.HorizontalFlip(),
            A.CLAHE(),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
            A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=30, val_shift_limit=5),
            A.ToFloat()
        ])
    elif transform_variant == 'distortions':
        transform = A.Compose([
            IAAPerspective2(scale=(0.0, 0.06)),
            IAAAffine2(scale=(0.7, 1.3),
                       rotate=(-40, 40),
                       shear=(-0.1, 0.1)),
            A.PadIfNeeded(min_height=out_size, min_width=out_size),
            A.OpticalDistortion(),
            A.RandomCrop(height=out_size, width=out_size),
            A.HorizontalFlip(),
            A.CLAHE(),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
            A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=30, val_shift_limit=5),
            A.ToFloat()
        ])
    elif transform_variant == 'distortions_scale05_1':
        transform = A.Compose([
            IAAPerspective2(scale=(0.0, 0.06)),
            IAAAffine2(scale=(0.5, 1.0),
                       rotate=(-40, 40),
                       shear=(-0.1, 0.1),
                       p=1),
            A.PadIfNeeded(min_height=out_size, min_width=out_size),
            A.OpticalDistortion(),
            A.RandomCrop(height=out_size, width=out_size),
            A.HorizontalFlip(),
            A.CLAHE(),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
            A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=30, val_shift_limit=5),
            A.ToFloat()
        ])
    elif transform_variant == 'distortions_scale03_12':
        transform = A.Compose([
            IAAPerspective2(scale=(0.0, 0.06)),
            IAAAffine2(scale=(0.3, 1.2),
                       rotate=(-40, 40),
                       shear=(-0.1, 0.1),
                       p=1),
            A.PadIfNeeded(min_height=out_size, min_width=out_size),
            A.OpticalDistortion(),
            A.RandomCrop(height=out_size, width=out_size),
            A.HorizontalFlip(),
            A.CLAHE(),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
            A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=30, val_shift_limit=5),
            A.ToFloat()
        ])
    elif transform_variant == 'distortions_scale03_07':
        transform = A.Compose([
            IAAPerspective2(scale=(0.0, 0.06)),
            IAAAffine2(scale=(0.3, 0.7),  # scale 512 to 256 in average
                       rotate=(-40, 40),
                       shear=(-0.1, 0.1),
                       p=1),
            A.PadIfNeeded(min_height=out_size, min_width=out_size),
            A.OpticalDistortion(),
            A.RandomCrop(height=out_size, width=out_size),
            A.HorizontalFlip(),
            A.CLAHE(),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
            A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=30, val_shift_limit=5),
            A.ToFloat()
        ])
    elif transform_variant == 'distortions_light':
        transform = A.Compose([
            IAAPerspective2(scale=(0.0, 0.02)),
            IAAAffine2(scale=(0.8, 1.8),
                       rotate=(-20, 20),
                       shear=(-0.03, 0.03)),
            A.PadIfNeeded(min_height=out_size, min_width=out_size),
            A.RandomCrop(height=out_size, width=out_size),
            A.HorizontalFlip(),
            A.CLAHE(),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
            A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=30, val_shift_limit=5),
            A.ToFloat()
        ])
    elif transform_variant == 'non_space_transform':
        transform = A.Compose([
            A.CLAHE(),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
            A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=30, val_shift_limit=5),
            A.ToFloat()
        ])
    elif transform_variant == 'no_augs':
        transform = A.Compose([
            A.ToFloat()
        ])
    else:
        raise ValueError(f'Unexpected transform_variant {transform_variant}')
    return transform


def make_default_train_dataloader(indir, kind='default', out_size=512, mask_gen_kwargs=None, transform_variant='default',
                                  mask_generator_kind="mixed", dataloader_kwargs=None, ddp_kwargs=None, **kwargs):
    LOGGER.info(f'Make train dataloader {kind} from {indir}. Using mask generator={mask_generator_kind}')

    mask_generator = get_mask_generator(kind=mask_generator_kind, kwargs=mask_gen_kwargs)
    transform = get_transforms(transform_variant, out_size)

    if kind == 'default':
        dataset = InpaintingTrainDataset(indir=indir,
                                         mask_generator=mask_generator,
                                         transform=transform,
                                         **kwargs)
    elif kind == 'default_web':
        dataset = InpaintingTrainWebDataset(indir=indir,
                                            mask_generator=mask_generator,
                                            transform=transform,
                                            **kwargs)
    elif kind == 'img_with_segm':
        dataset = ImgSegmentationDataset(indir=indir,
                                         mask_generator=mask_generator,
                                         transform=transform,
                                         out_size=out_size,
                                         **kwargs)
    else:
        raise ValueError(f'Unknown train dataset kind {kind}')

    if dataloader_kwargs is None:
        dataloader_kwargs = {}

    is_dataset_only_iterable = kind in ('default_web',)

    if ddp_kwargs is not None and not is_dataset_only_iterable:
        dataloader_kwargs['shuffle'] = False
        dataloader_kwargs['sampler'] = DistributedSampler(dataset, **ddp_kwargs)

    if is_dataset_only_iterable and 'shuffle' in dataloader_kwargs:
        with open_dict(dataloader_kwargs):
            del dataloader_kwargs['shuffle']

    dataloader = DataLoader(dataset, **dataloader_kwargs)
    return dataloader


def make_default_val_dataset(indir, kind='default', out_size=512, transform_variant='default', **kwargs):
    if OmegaConf.is_list(indir) or isinstance(indir, (tuple, list)):
        return ConcatDataset([
            make_default_val_dataset(idir, kind=kind, out_size=out_size, transform_variant=transform_variant, **kwargs) for idir in indir 
        ])

    LOGGER.info(f'Make val dataloader {kind} from {indir}')
    mask_generator = get_mask_generator(kind=kwargs.get("mask_generator_kind"), kwargs=kwargs.get("mask_gen_kwargs"))

    if transform_variant is not None:
        transform = get_transforms(transform_variant, out_size)

    if kind == 'default':
        dataset = InpaintingEvaluationDataset(indir, **kwargs)
    elif kind == 'our_eval':
        dataset = OurInpaintingEvaluationDataset(indir, **kwargs)
    elif kind == 'img_with_segm':
        dataset = ImgSegmentationDataset(indir=indir,
                                         mask_generator=mask_generator,
                                         transform=transform,
                                         out_size=out_size,
                                         **kwargs)
    elif kind == 'online':
        dataset = InpaintingEvalOnlineDataset(indir=indir,
                                              mask_generator=mask_generator,
                                              transform=transform,
                                              out_size=out_size,
                                              **kwargs)
    else:
        raise ValueError(f'Unknown val dataset kind {kind}')

    return dataset


def make_default_val_dataloader(*args, dataloader_kwargs=None, **kwargs):
    dataset = make_default_val_dataset(*args, **kwargs)

    if dataloader_kwargs is None:
        dataloader_kwargs = {}
    dataloader = DataLoader(dataset, **dataloader_kwargs)
    return dataloader


def make_constant_area_crop_params(img_height, img_width, min_size=128, max_size=512, area=256*256, round_to_mod=16):
    min_size = min(img_height, img_width, min_size)
    max_size = min(img_height, img_width, max_size)
    if random.random() < 0.5:
        out_height = min(max_size, ceil_modulo(random.randint(min_size, max_size), round_to_mod))
        out_width = min(max_size, ceil_modulo(area // out_height, round_to_mod))
    else:
        out_width = min(max_size, ceil_modulo(random.randint(min_size, max_size), round_to_mod))
        out_height = min(max_size, ceil_modulo(area // out_width, round_to_mod))

    start_y = random.randint(0, img_height - out_height)
    start_x = random.randint(0, img_width - out_width)
    return (start_y, start_x, out_height, out_width)
