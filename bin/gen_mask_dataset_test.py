#!/usr/bin/env python3

import glob
import os
import shutil
import traceback

import PIL.Image as Image
import numpy as np
from joblib import Parallel, delayed

from saicinpainting.evaluation.masks.mask import SegmentationMask, propose_random_square_crop
from saicinpainting.evaluation.utils import load_yaml, SmallMode
from saicinpainting.training.data.masks import MixedMaskGenerator

import sys

import os.path as osp
import scipy
import scipy.io as io
from scipy.io import loadmat , savemat

import PIL
from PIL import Image

import mask_k


def k2wgt(X,W):
    result = np.multiply(X,W)
    return result


def write_images(x,image_save_path):
    x = np.clip(x * 255, 0, 255).astype(np.uint8)
    cv2.imwrite(image_save_path, x)

class MakeManyMasksWrapper:
    def __init__(self, impl, variants_n=2):
        self.impl = impl
        self.variants_n = variants_n

    def get_masks(self, img):
        img = np.transpose(np.array(img), (2, 0, 1))
        return [self.impl(img)[0] for _ in range(self.variants_n)]


def process_images(src_images, indir, outdir, config):
    if config.generator_kind == 'segmentation':
        mask_generator = SegmentationMask(**config.mask_generator_kwargs)
    elif config.generator_kind == 'random':
        variants_n = config.mask_generator_kwargs.pop('variants_n', 2)
        mask_generator = MakeManyMasksWrapper(MixedMaskGenerator(**config.mask_generator_kwargs),
                                              variants_n=variants_n)
    else:
        raise ValueError(f'Unexpected generator kind: {config.generator_kind}')

    max_tamper_area = config.get('max_tamper_area', 1)

    for infile in src_images:
        try:
            file_relpath = infile[len(indir):]
            img_outpath = os.path.join(outdir, file_relpath)
            os.makedirs(os.path.dirname(img_outpath), exist_ok=True)

            #image = Image.open(infile).convert('RGB')
            #print("  image.type=",type(image)," shape =",np.array(image).shape)
            print("path=",infile)
            file_name = os.path.basename(infile)
            print("file_name=",file_name)


            

            #将数据读取出来
            siat_input =scipy.io.loadmat(infile)
            siat_input = siat_input['Img2']   #256x256x3
            print(type(siat_input),siat_input.shape)
            

            #转为k空间
            siat = np.array(siat_input[:,: , 0:2], dtype=np.float32)
            #print(siat.shape)
            siat_complex = siat[:, :, 0] + 1j * siat[:, :, 1]
            #print("siat_complex.shape=",siat_complex.shape)
            siat_kdata = np.fft.fft2(siat_complex)
            siat_kdata = np.fft.fftshift(siat_kdata)
            # data = {"kdata": siat_kdata}
            # savemat("/home/b109/code/xx/DiffIR-master/DiffIR-master/DiffIR-inpainting/data/mri4000/sait_kdata/"+file_name , data)

            #加载权重
            #weight = loadmat('/home/b109/Desktop/XX/inpainting/data/mri/weight/weight1_GEBrain.mat')['weight']
            weight = loadmat('/home/b109/Desktop/XX/inpainting/data/mri/weight/weight1_PeiBrain.mat')['weight']
            kdata_w = k2wgt(siat_kdata, weight)

            data={"k_w":kdata_w}
            savemat("/home/b109/Desktop/XX/inpainting/data/temp/kw_test.mat" , data)

            #重新转为3x256x256
            siat_temp = np.zeros((256, 256, 3))
            kdata = np.array(siat_temp, dtype=np.float32)
            kdata[:, :, 0] = np.real(kdata_w)
            kdata[:, :, 1] = np.imag(kdata_w)
            kdata[:, :, 2] = np.real(kdata_w)
            #kdata = kdata.transpose((2, 0, 1))      # 3x256x256

            # # scale input image to output resolution and filter smaller images
            # if min(image.size) < config.cropping.out_min_size:
            #     handle_small_mode = SmallMode(config.cropping.handle_small_mode)
            #     if handle_small_mode == SmallMode.DROP:
            #         continue
            #     elif handle_small_mode == SmallMode.UPSCALE:
            #         factor = config.cropping.out_min_size / min(image.size)
            #         out_size = (np.array(image.size) * factor).round().astype('uint32')
            #         image = image.resize(out_size, resample=Image.BICUBIC)
            # else:
            #     factor = config.cropping.out_min_size / min(image.size)
            #     out_size = (np.array(image.size) * factor).round().astype('uint32')
            #     image = image.resize(out_size, resample=Image.BICUBIC)

            # generate and select masks
            #image = PIL.Image.fromarray(kdata)
            #src_masks = mask_generator.get_masks(image)
            

            #固定mask
            mask = np.zeros((1, 256, 256))
            #mat=io.loadmat('/home/b109/code/xx/DiffIR-master/DiffIR-master/DiffIR-inpainting/data/mri4000/mask/random/random_0.1.mat')
            mat=io.loadmat('/home/b109/Desktop/XX/inpainting/data/mri/mask/poisson/6.mat')
            #mat=io.loadmat('/home/b109/code/xx/DiffIR-master/DiffIR-master/DiffIR-inpainting/data/mask/cart/mask_cart_030.mat')
            mask_item = mat['mask']
            mask[0, :, :] = mask_item
            #mask=1-mask
            src_masks=mask
            #write_images(abs(mask_item), osp.join('/home/b109/code/xx/DiffIR-master/DiffIR-master/DiffIR-inpainting/data/temp/mask.png'))


            #不固定
            #加载mask
            # mask = np.zeros((1, 256, 256))


            # mask[0, :, :] = mask_k.multiply_random_mask_k_space(kdata_w) 
            # mask = 1-mask
            
            # # src_masks = mask_generator.get_masks(kdata)
            # src_masks=mask
            image = kdata
            

            filtered_image_mask_pairs = []
            for cur_mask in src_masks:
                if config.cropping.out_square_crop:
                    (crop_left,
                     crop_top,
                     crop_right,
                     crop_bottom) = propose_random_square_crop(cur_mask,
                                                               min_overlap=config.cropping.crop_min_overlap)
                    cur_mask = cur_mask[crop_top:crop_bottom, crop_left:crop_right]
                    #cur_image = image.copy().crop((crop_left, crop_top, crop_right, crop_bottom))
                    cur_image = image
                else:
                    cur_image = image

                if len(np.unique(cur_mask)) == 0 or cur_mask.mean() > max_tamper_area:
                    continue

                filtered_image_mask_pairs.append((cur_image, cur_mask))

            mask_indices = np.random.choice(len(filtered_image_mask_pairs),
                                            size=min(len(filtered_image_mask_pairs), config.max_masks_per_image),
                                            replace=False)

            # crop masks; save masks together with input image
            mask_basename = os.path.join(outdir, os.path.splitext(file_relpath)[0])
            for i, idx in enumerate(mask_indices):
                cur_image, cur_mask = filtered_image_mask_pairs[idx]
                cur_basename = mask_basename + f'_crop{i:03d}'
                Image.fromarray(np.clip(cur_mask * 255, 0, 255).astype('uint8'),
                                mode='L').save(cur_basename + f'_mask{i:03d}.png')
                mask={"mask":np.clip(cur_mask * 255, 0, 255).astype('uint8')}
                savemat("/home/b109/Desktop/XX/inpainting/data/mri/val_256/random_thick_256_test/"+"_mask"+file_name, mask)
                # cur_image.save(cur_basename + '.mat')
                data={"k_w":cur_image}
                print(9999)
                savemat("/home/b109/Desktop/XX/inpainting/data/mri/val_256/random_thick_256_test/"+file_name, data)

        except KeyboardInterrupt:
            return
        except Exception as ex:
            print(f'Could not make masks for {infile} due to {ex}:\n{traceback.format_exc()}')


def main(args):
    if not args.indir.endswith('/'):
        args.indir += '/'

    os.makedirs(args.outdir, exist_ok=True)

    config = load_yaml(args.config)


    in_files = list(glob.glob(os.path.join(args.indir, '**', f'*.{args.ext}'), recursive=True))

    if args.n_jobs == 0:
        process_images(in_files, args.indir, args.outdir, config)
    else:
        in_files_n = len(in_files)
        chunk_size = in_files_n // args.n_jobs + (1 if in_files_n % args.n_jobs > 0 else 0)
        Parallel(n_jobs=args.n_jobs)(
            delayed(process_images)(in_files[start:start+chunk_size], args.indir, args.outdir, config)
            for start in range(0, len(in_files), chunk_size)
        )


if __name__ == '__main__':
    import argparse

    aparser = argparse.ArgumentParser()
    aparser.add_argument('config', type=str, help='Path to config for dataset generation')
    aparser.add_argument('indir', type=str, help='Path to folder with images')
    aparser.add_argument('outdir', type=str, help='Path to folder to store aligned images and masks to')
    aparser.add_argument('--n-jobs', type=int, default=0, help='How many processes to use')
    aparser.add_argument('--ext', type=str, default='mat', help='Input image extension')

    main(aparser.parse_args())
