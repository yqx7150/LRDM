#!/usr/bin/env python3

# Example command:
# ./bin/predict.py \
#       model.path=<path to checkpoint, prepared by make_checkpoint.py> \
#       indir=<path to input data> \
#       outdir=<where to store predicts>

import logging
import os
import sys
import traceback

from saicinpainting.evaluation.utils import move_to_device
from saicinpainting.evaluation.refinement import refine_predict
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import cv2
import hydra
import numpy as np
import torch
import tqdm
import yaml
from omegaconf import OmegaConf
from torch.utils.data._utils.collate import default_collate

from saicinpainting.training.data.datasets import make_default_val_dataset
from saicinpainting.training.trainers import load_checkpoint
from saicinpainting.utils import register_debug_signal_handlers




from pprint import pprint
import os.path as osp
import scipy
import scipy.io as io
from scipy.io import loadmat , savemat
import sys
import PIL.Image as Image
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import mean_squared_error as compare_mse


LOGGER = logging.getLogger(__name__)


def fft2c_noshift(x):
    size = (x).shape
    fctr = size[0]*size[1]
    Kdata = np.zeros((size),dtype=np.complex64)
    for i in range(size[2]):
        Kdata[:,:,i] = (1/np.sqrt(fctr))*np.fft.fft2(x[:,:,i])
    return Kdata

def write_kdata(Kdata,name,path):
    temp = np.log(1+abs(Kdata))    
    plt.axis('off')
    plt.imshow(abs(temp),cmap='gray')
    plt.savefig(osp.join(path,name),transparent=True, dpi=128, pad_inches = 0,bbox_inches = 'tight')

def write_Data(model_num,psnr,ssim,name,path):
    filedir=name+"result.txt"
    with open(osp.join(path,filedir),"w+") as f:#a+
        f.writelines(str(model_num)+' '+'['+str(round(psnr, 2))+' '+str(round(ssim, 4))+']')
        f.write('\n')
        
def write_Data2(psnr,ssim,name,path):
    filedir=name+"PC_SAKE.txt"
    with open(osp.join(path,filedir),"a+") as f:#a+
        f.writelines('['+str(round(psnr, 2))+' '+str(round(ssim, 4))+']')
        f.write('\n')
        
def write_images(x,image_save_path):
    x = np.clip(x * 255, 0, 255).astype(np.uint8)
    cv2.imwrite(image_save_path, x)
  
def k2wgt(X,W):
    Y = np.multiply(X,W) 
    return Y

def wgt2k(X,W,DC):
    Y = np.multiply(X,1./W)
    Y[W==0] = DC[W==0] 
    return Y


@hydra.main(config_path='../configs/prediction', config_name='default.yaml')
def main(predict_config: OmegaConf):
    try:
        register_debug_signal_handlers()  # kill -10 <pid> will result in traceback dumped into log

        device = torch.device(predict_config.device)

        train_config_path = os.path.join(predict_config.model.path, 'config.yaml')
        with open(train_config_path, 'r') as f:
            train_config = OmegaConf.create(yaml.safe_load(f))
        
        train_config.training_model.predict_only = True
        train_config.visualizer.kind = 'noop'

        out_ext = predict_config.get('out_ext', '.png')


        checkpoint_path = os.path.join(predict_config.model.path, 
                                       'models', 
                                       predict_config.model.checkpoint)
        model = load_checkpoint(train_config, checkpoint_path, strict=False, map_location='cpu')
        model.freeze()
        if not predict_config.get('refine', False):
            model.to(device)

        if not predict_config.indir.endswith('/'):
            predict_config.indir += '/'

        #dataset = make_default_val_dataset(predict_config.indir, **predict_config.dataset)
        #save_path = "/home/b109/code/xx/DiffIR-master/DiffIR-master/DiffIR-inpainting/inference/mri_random_thick_256_T/"











        save_path=predict_config.outdir
        coil = 8

        #读取一个多线圈数据
        ori_input = np.zeros([256, 256, coil], dtype=np.complex64)
        #ori_input =scipy.io.loadmat("/home/b109/code/xx/DiffIR-master/DiffIR-master/DiffIR-inpainting/data/mri4000/val_256/random_thick_256_test/334.mat")['Img2']
        #ori_input =scipy.io.loadmat("/home/b109/code/xx/DiffIR-master/DiffIR-master/DiffIR-inpainting/data/mri4000/test_fastMRI_no_img_kdata/3.mat")['Img2']  #3
        ori_input =scipy.io.loadmat("/home/b109/code/xx/DiffIR-master/DiffIR-master/DiffIR-inpainting/data/mri4000/val_256/random_thick_256_test/data1_GE_brain.mat")['DATA']
        print("ori_input.shape=",ori_input.shape)
        ori_input = ori_input / np.max(abs(ori_input))
        ori_data_sos = np.sqrt(np.sum(np.square(np.abs(ori_input)),axis=2)) 
        write_images(abs(ori_data_sos),osp.join(save_path,'ori'+'.png'))
        io.savemat(osp.join(save_path,'ori_data_sos.mat'),{'ori_data_sos':ori_data_sos})



      
        #读取权重
        #ww = io.loadmat('/home/b109/code/xx/DiffIR-master/DiffIR-master/DiffIR-inpainting/data/mri4000/weight/weight1_PeiBrain.mat')['weight']
        #ww = io.loadmat('/home/b109/code/xx/DiffIR-master/DiffIR-master/DiffIR-inpainting/data/mri4000/weight/weight1_GEBrain.mat')['weight']   #38.24
        #ww = io.loadmat('/home/b109/code/xx/DiffIR-master/DiffIR-master/DiffIR-inpainting/data/mri4000/weight/weight_GE_brain_07.mat')['weight']   #38.74
        ww = io.loadmat('/home/b109/code/xx/DiffIR-master/DiffIR-master/DiffIR-inpainting/data/mri4000/weight/weight_GE_brain_07.mat')['weight']   #09 36.931
        #ww = io.loadmat('/home/b109/code/xx/DiffIR-master/DiffIR-master/DiffIR-inpainting/data/mri4000/weight/weight_GE_brain_test.mat')['weight']
        weight = np.zeros((coil,256,256))       
        for i in range(coil):
            weight[i,:,:] = ww

            
                    
        

        #读取mask
        #======== mask
        mask = np.zeros((coil*2,256,256))
        mask_item = io.loadmat('/home/b109/code/xx/DiffIR-master/DiffIR-master/DiffIR-inpainting/data/mri4000/mask/poisson/6.mat')['mask']
        #mask_item = io.loadmat('/home/b109/code/xx/DiffIR-master/DiffIR-master/DiffIR-inpainting/data/mri4000/mask/me/mask2.mat')['mask']
        #mask_item=io.loadmat('/home/b109/code/xx/DiffIR-master/DiffIR-master/DiffIR-inpainting/data/mask/cart/mask_cart_010.mat')['mask']
        print("mask_item.shape=",mask_item.shape)
        for i in range(coil*2):
            mask[i,:,:] = mask_item
        mask2=mask
        mask2_img = np.sqrt(np.sum(np.square(np.abs(mask2)),axis=0))
        mask2_img = mask2_img/np.max(np.abs(mask2_img))
        write_images(abs(mask2_img),osp.join(save_path,'mask2_img'+'.png'))
        io.savemat(osp.join(save_path,'mask.mat'),{'mask':mask})

        #零填充
        Kdata = np.zeros((coil, 256, 256), dtype=np.complex64)
        Ksample = np.zeros((coil,256,256),dtype=np.complex64)
        zeorfilled_data = np.zeros((coil,256,256),dtype=np.complex64)
        for i in range(coil):
            Kdata[i, :, :] = np.fft.fft2(ori_input[:, :,i])
            Kdata[i, :, :] = np.fft.fftshift(Kdata[i, :, :])
            #Kdata[i, :, :] = Kdata[i, :, :]/np.max(np.abs(Kdata[i, :, :]))
            Ksample[i,:,:] = np.multiply(mask[i,:,:],Kdata[i,:,:])  
            zeorfilled_data[i,:,:] = np.fft.ifft2(Ksample[i,:,:])  
        

        zeorfilled_data_sos = np.sqrt(np.sum(np.square(np.abs(zeorfilled_data)),axis=0))
        ori_data_sos = ori_data_sos/np.max(np.abs(ori_data_sos))
        zeorfilled_data_sos = zeorfilled_data_sos/np.max(np.abs(zeorfilled_data_sos))  
        print("zeorfilled_data_sos.shape=",zeorfilled_data_sos.shape,"   ori_data_sos.shape=",ori_data_sos.shape)

        psnr_zero=compare_psnr(255*abs(zeorfilled_data_sos),255*abs(ori_data_sos),data_range=255)
        ssim_zero=compare_ssim(abs(zeorfilled_data_sos),abs(ori_data_sos),data_range=1)
        print('psnr_zero: ',psnr_zero,'ssim_zero: ',ssim_zero)
        write_images(abs(zeorfilled_data_sos),osp.join(save_path,'Zeorfilled_'+'.png'))
        io.savemat(osp.join(save_path,'zeorfilled.mat'),{'zeorfilled':zeorfilled_data})
        io.savemat(osp.join(save_path,'Ksample.mat'),{'Ksample':Ksample})
        io.savemat(osp.join(save_path,'Kdata.mat'),{'Kdata':Kdata})



        #乘权重
        k_w = np.zeros((coil,256,256),dtype=np.complex64)
        x_input=np.random.uniform(-1, 1, size=(coil*2, 3, 256, 256))
        for i in range(coil):
            k_w[i,:,:] = k2wgt(Kdata[i,:,:],weight[i,:,:])  
            x_input[i,0,:,:] =  np.real(k_w[i,:,:])
            x_input[i,1,:,:] =  np.imag(k_w[i,:,:])
            x_input[i,2,:,:] =  np.real(k_w[i,:,:])

            x_input[i+coil,0,:,:] =  np.real(k_w[i,:,:])
            x_input[i+coil,1,:,:] =  np.imag(k_w[i,:,:])
            x_input[i+coil,2,:,:] =  np.real(k_w[i,:,:])

        print("max = ",np.max(x_input),"   min=",np.min(x_input))
        
 


        io.savemat(osp.join(save_path,'k_w.mat'),{'k_w':k_w[0,:,:]})


        x_input = x_input.astype(np.float32)
        x_input = torch.from_numpy(x_input)
        print("x_input.shape=",x_input.shape, "  type=",type(x_input))


        

        Kdata_sos = np.sqrt(np.sum(np.square(np.abs(Kdata)),axis=0))
        Kdata_sos = Kdata_sos/np.max(np.abs(Kdata_sos))
        write_images(abs(Kdata_sos),osp.join(save_path,'Kdata_sos'+'.png'))

        k_w_sos = np.sqrt(np.sum(np.square(np.abs(k_w)),axis=0))
        k_w_sos = k_w_sos/np.max(np.abs(k_w_sos))
        write_images(abs(k_w_sos),osp.join(save_path,'k_w_sos'+'.png'))

        mask=1-mask
        mask = mask.astype(np.float32)
        mask_tensor = torch.from_numpy(mask)
        mask = mask_tensor.unsqueeze(1)
        print("mask.shape=",mask.shape,"  type=",type(mask))


        #将这个x_input送入batch
        #将mask送入batch
        batch = {"mask": mask, "image": x_input}
        batch = move_to_device(batch, device)
        #送入模型进行计算
        batch = model(batch)

        result=batch[predict_config.out_key]
        print("result.shape=",result.shape)

        rec_k_w = np.zeros((coil,256,256),dtype=np.complex64)
        kw_real = np.zeros((coil,256,256),dtype=np.float32)
        kw_imag = np.zeros((coil,256,256),dtype=np.float32)  
        #读取结果
        for i in range(coil):
            cur_res_1 = batch[predict_config.out_key][i].permute(1, 2, 0).detach().cpu().numpy()
            cur_res_2 = batch[predict_config.out_key][i+coil].permute(1, 2, 0).detach().cpu().numpy()
            cur_res=(cur_res_1+cur_res_2)/2
            kw_real[i,:,:] = cur_res[:,:,0]
            kw_imag[i,:,:] = cur_res[:,:,1]
            rec_k_w[i,:,:] = kw_real[i,:,:]+1j*kw_imag[i,:,:]
            rec_k_w_i = rec_k_w[i,:,:]
            rec_k_w_i = rec_k_w_i/np.max(np.abs(rec_k_w_i))
            write_images(abs(rec_k_w_i),osp.join(save_path,'rec_k_w_'+str(i)+'.png'))

        print("cur_res_max = ",np.max(cur_res),"   rec_k_w_min=",np.min(cur_res))
        io.savemat(osp.join(save_path,'rec_k_w.mat'),{'rec_k_w':rec_k_w[0,:,:]})
        
        #算rec后的结果  
        k_complex = np.zeros((coil,256,256),dtype=np.complex64)
        k_complex2 = np.zeros((coil,256,256),dtype=np.complex64)
        NCSNpp_Image = np.zeros((coil,256,256),dtype=np.complex64)
        NCSNpp_Image_noDC = np.zeros((coil,256,256),dtype=np.complex64)
            #Ksample = np.fft.fftshift(Ksample)

        #除去权重+DC
        for i in range(coil):
            k_complex[i,:,:] = wgt2k(rec_k_w[i,:,:],weight[i,:,:],Ksample[i,:,:])
            NCSNpp_Image_noDC[i,:,:] = np.fft.ifft2(k_complex[i,:,:])
            k_complex2[i,:,:] = Ksample[i,:,:] + k_complex[i,:,:]*(1-mask2[i,:,:])
            NCSNpp_Image[i,:,:] = np.fft.ifft2(k_complex2[i,:,:])
            NCSNpp_Image_i = NCSNpp_Image[i,:,:]
            NCSNpp_Image_i = NCSNpp_Image_i/np.max(np.abs(NCSNpp_Image_i))
            write_images(abs(NCSNpp_Image_i),osp.join(save_path,'NCSNpp_Image_'+str(i)+'.png'))

        print("k_complex_max = ",np.max(k_complex),"   k_complex_min=",np.min(k_complex))
        print("k_complex2_max = ",np.max(k_complex2),"   k_complex2_min=",np.min(k_complex2))
        io.savemat(osp.join(save_path,'k_complex2.mat'),{'k_complex2':k_complex2[0,:,:]})
        io.savemat(osp.join(save_path,'k_complex.mat'),{'k_complex':k_complex[0,:,:]})

        Ksample_sos = np.sqrt(np.sum(np.square(np.abs(Ksample)),axis=0))
        Ksample_sos = Ksample_sos/np.max(np.abs(Ksample_sos))
        write_images(abs(Ksample_sos),osp.join(save_path,'Ksample_sos'+'.png'))

        rec_k_w_sos = np.sqrt(np.sum(np.square(np.abs(rec_k_w)),axis=0))
        rec_k_w_sos = rec_k_w_sos/np.max(np.abs(rec_k_w_sos))
        write_images(abs(rec_k_w_sos),osp.join(save_path,'rec_k_w_sos'+'.png'))


        rec_k_complex_sos = np.sqrt(np.sum(np.square(np.abs(k_complex2)),axis=0))
        rec_k_complex_sos = rec_k_complex_sos/np.max(np.abs(rec_k_complex_sos))
        write_images(abs(rec_k_complex_sos),osp.join(save_path,'rec_k_complex_sos'+'.png'))



        NCSNpp_Image_noDC_sos = np.sqrt(np.sum(np.square(np.abs(NCSNpp_Image_noDC)),axis=0))
        NCSNpp_Image_noDC_sos = NCSNpp_Image_noDC_sos/np.max(np.abs(NCSNpp_Image_noDC_sos))
        write_images(abs(NCSNpp_Image_noDC_sos),osp.join(save_path,'Rec_noDC'+'.png'))

        rec_Image_sos = np.sqrt(np.sum(np.square(np.abs(NCSNpp_Image)),axis=0))
        rec_Image_sos = rec_Image_sos/np.max(np.abs(rec_Image_sos))

            # Print PSNR
        print("NCSNpp_Image_noDC_sos.shape=",NCSNpp_Image_noDC_sos.shape ,"  ori_data_sos.shape=",ori_data_sos.shape)
        psnr = compare_psnr(255*abs(NCSNpp_Image_noDC_sos),255*abs(ori_data_sos),data_range=255)
        ssim = compare_ssim(abs(NCSNpp_Image_noDC_sos),abs(ori_data_sos),data_range=1)
        print(' PSNR_noDC:', psnr,' SSIM_noDC:', ssim)  


        psnr = compare_psnr(255*abs(rec_Image_sos),255*abs(ori_data_sos),data_range=255)
        ssim = compare_ssim(abs(rec_Image_sos),abs(ori_data_sos),data_range=1)
        print(' PSNR:', psnr,' SSIM:', ssim)  
        write_images(abs(rec_Image_sos),osp.join(save_path,'Rec'+'.png'))



        sys.exit()
            #sys.exit()

    except KeyboardInterrupt:
        LOGGER.warning('Interrupted by user')
    except Exception as ex:
        LOGGER.critical(f'Prediction failed due to {ex}:\n{traceback.format_exc()}')
        sys.exit(1)


if __name__ == '__main__':
    main()
