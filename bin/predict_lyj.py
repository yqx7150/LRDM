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
            train_config = OmegaConf.create(yFaml.safe_load(f))
        
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

        dataset = make_default_val_dataset(predict_config.indir, **predict_config.dataset)
        #save_path = "/home/b109/code/xx/DiffIR-master/DiffIR-master/DiffIR-inpainting/inference/mri_random_thick_256_T/"
        save_path=predict_config.outdir
        coil=1
        for img_i in tqdm.trange(len(dataset)):
            mask_fname = dataset.mask_filenames[img_i]
            cur_out_fname = os.path.join(
                predict_config.outdir, 
                os.path.splitext(mask_fname[len(predict_config.indir):])[0] + out_ext
            )
            os.makedirs(os.path.dirname(cur_out_fname), exist_ok=True)
            batch = default_collate([dataset[img_i]])
            if predict_config.get('refine', False):
                assert 'unpad_to_size' in batch, "Unpadded size is required for the refinement"
                # image unpadding is taken care of in the refiner, so that output image
                # is same size as the input image
                cur_res = refine_predict(batch, model, **predict_config.refiner)
                cur_res = cur_res[0].permute(1,2,0).detach().cpu().numpy()
            else:
                with torch.no_grad():
                    batch = move_to_device(batch, device)
                    batch['mask'] = (batch['mask'] > 0) * 1
                    print("type(mask)=",type(batch['mask']),"   shape =",batch['mask'].shape)
                    #k_w的三维数据
                    print("type(image)=",type(batch['image']),"   shape =",batch['image'].shape)



                    #读取一个多线圈数据
                    # ori_input =scipy.io.loadmat("/home/b109/code/xx/DiffIR-master/DiffIR-master/DiffIR-inpainting/data/mri4000/val_256/random_thick_256_test/data1_GE_brain.mat")['DATA']
                    # print("ori_input.shape=",ori_input.shape)
                    # coil = 8
      

                    # ww = io.loadmat('/home/b109/code/xx/DiffIR-master/DiffIR-master/DiffIR-inpainting/data/mri4000/weight/weight1.mat')['weight']
                    # weight = np.zeros((coil,256,256))       
                    # for i in range(coil):
                    #     weight[i,:,:] = ww
                    
                    # ori_input=ori_input.reshape(coil,256,256)
                    # Kdata = np.zeros((coil,256,256),dtype=np.complex64)
                    # k_w = np.zeros((coil,256,256),dtype=np.complex64)
                    # for i in range(coil):
                    #     Kdata[i,:,:] = np.fft.fft2(ori_input[i,:,:])# max: 3820.8044
                    #     k_w[i,:,:] = k2wgt(Kdata[i,:,:],weight[i,:,:]) # max: 0.42637014  
                    # print("k_w.shape=",k_w.shape) 
                             
                    #将k_w放入batch['image']


                    #sys.exit()
                    
                    #sys.exit()
                    batch = model(batch)                    
                    cur_res = batch[predict_config.out_key][0].permute(1, 2, 0).detach().cpu().numpy()
                    print("type(cur_res)=",type(cur_res),"   shape =",cur_res.shape)
                    #sys.exit()
                    unpad_to_size = batch.get('unpad_to_size', None)
                    if unpad_to_size is not None:
                        orig_height, orig_width = unpad_to_size
                        cur_res = cur_res[:orig_height, :orig_width]
            


            #cur_res = np.clip(cur_res * 255, 0, 255).astype('uint8')
            #cur_res = cv2.cvtColor(cur_res, cv2.COLOR_RGB2BGR)
            #cv2.imwrite(cur_out_fname, cur_res)

            #处理保存恢复的图像
            # data={"k_w_rec":cur_res}
            # savemat("/home/b109/code/xx/DiffIR-master/DiffIR-master/DiffIR-inpainting/inference/mri_random_thick_256_T/"+"k_w_rec.mat", data)

            #除去权重并DC
            
            #将数据读取出来
            number=mask_fname.split('/')[-1].split('_')[0]
            img_name=number+"_"
            print("img_name=",img_name)
            siat_input =scipy.io.loadmat("/home/b109/code/xx/DiffIR-master/DiffIR-master/DiffIR-inpainting/data/mri4000/val_256/val_256_zl/data/5.mat")
            # siat_input =scipy.io.loadmat("/home/b109/code/xx/DiffIR-master/DiffIR-master/DiffIR-inpainting/data/mri4000/val_256/val_256_zl/"+number+".mat")
            # assert 0
            siat_input = siat_input['Img2']   #256x256x3
            #print(type(siat_input),siat_input.shape)

            #转为k空间
            siat = np.array(siat_input[:,: , 0:2], dtype=np.float32)
            #print(siat.shape)
            siat_complex_ori = siat[:, :, 0] + 1j * siat[:, :, 1]
            print("siat_complex_ori.shape=",siat_complex_ori.shape)
            siat_kdata = np.fft.fft2(siat_complex_ori)
            siat_kdata = np.fft.fftshift(siat_kdata)

            write_images(abs(siat_complex_ori),osp.join(save_path,number+'ori'+'.png'))

            
            #加载权重
            weight = np.zeros((coil,256,256))  
            ww = loadmat('/home/b109/code/xx/DiffIR-master/DiffIR-master/DiffIR-inpainting/data/mri4000/weight/weight1.mat')['weight']
            for i in range(coil):
                weight[i,:,:] = ww
                
            #======== mask
            mask = np.zeros((coil,256,256))
            # mask_item = io.loadmat('/home/b109/code/xx/DiffIR-master/DiffIR-master/DiffIR-inpainting/data/mri4000/val_256/random_thick_256_test/_mask'+number+'.mat')['mask']
            mask_item = io.loadmat('/home/b109/code/xx/DiffIR-master/DiffIR-master/DiffIR-inpainting/data/mri4000/val_256/val_256_zl/mask/_mask5.mat')['mask']
            print("mask_item.shape=",mask_item.shape)
            for i in range(coil):
                mask[i,:,:] = mask_item
            mask = mask / 255
            
            print(np.sum(mask_item)/65536)
            #plt.imshow(abs(mask),cmap='gray')
            #plt.show()
            Image.fromarray(mask_item,
                                mode='L').save(save_path+number+'mask2'+'.png')
            


            #-------------------零填充----------------------------------------------------------
            #siat_complex_ori=siat_complex_ori/np.max(np.abs(siat_complex_ori))
            Kdata = np.zeros((coil,256,256),dtype=np.complex64)
            Ksample = np.zeros((coil,256,256),dtype=np.complex64)
            zeorfilled_data = np.zeros((coil,256,256),dtype=np.complex64)



            for i in range(coil):
                Kdata[i,:,:] = np.fft.fft2(siat_complex_ori)# max: 3820.8044
                Kdata[i,:,:]=np.fft.fftshift(Kdata[i,:,:])
                Ksample[i,:,:] = np.multiply(mask[i,:,:],Kdata[i,:,:]) # max: 3820.8044
                zeorfilled_data[i,:,:] = np.fft.ifft2(Ksample[i,:,:])  
            


            
            Kdata_sos = np.sqrt(np.sum(np.square(np.abs(Kdata)),axis=0))
            Kdata_sos = Kdata_sos/np.max(np.abs(Kdata_sos))
            write_images(abs(Kdata_sos),osp.join(save_path,img_name+'kdata'+'.png'))

            ksample_sos = np.sqrt(np.sum(np.square(np.abs(Ksample)),axis=0))
            ksample_sos = ksample_sos/np.max(np.abs(ksample_sos))
            write_images(abs(ksample_sos),osp.join(save_path,img_name+'ksample'+'.png'))
            
            zeorfilled_data_sos = np.sqrt(np.sum(np.square(np.abs(zeorfilled_data)),axis=0))
            ori_data_sos = siat_complex_ori

            zeorfilled_data_sos = zeorfilled_data_sos/np.max(np.abs(zeorfilled_data_sos))  

            psnr_zero=compare_psnr(255*abs(zeorfilled_data_sos),255*abs(ori_data_sos),data_range=255)
            ssim_zero=compare_ssim(abs(zeorfilled_data_sos),abs(ori_data_sos),data_range=1)
            print('psnr_zero: ',psnr_zero,'ssim_zero: ',ssim_zero)
            write_images(abs(zeorfilled_data_sos),osp.join(save_path,img_name+'Zeorfilled_'+'.png'))
            io.savemat(osp.join(save_path,img_name+'zeorfilled.mat'),{'zeorfilled':zeorfilled_data})

            #恢复成加权复数
            k_w = np.zeros((coil,256,256),dtype=np.complex64)
            kw_real = np.zeros((coil,256,256),dtype=np.float32)
            kw_imag = np.zeros((coil,256,256),dtype=np.float32)   
            for i in range(coil):    
                kw_real[i,:,:] = cur_res[:,:,0]
                kw_imag[i,:,:] = cur_res[:,:,1]
                k_w[i,:,:] = kw_real[i,:,:]+1j*kw_imag[i,:,:]





            #算rec后的结果  
            k_complex = np.zeros((coil,256,256),dtype=np.complex64)
            k_complex2 = np.zeros((coil,256,256),dtype=np.complex64)
            NCSNpp_Image = np.zeros((coil,256,256),dtype=np.complex64)
            Image_NoDC = np.zeros((coil, 256, 256), dtype=np.complex64)
            #Ksample = np.fft.fftshift(Ksample)
            for i in range(coil):
                k_complex[i,:,:] = wgt2k(k_w[i,:,:],weight[i,:,:],Ksample[i,:,:])
                k_complex2[i,:,:] = Ksample[i,:,:] + k_complex[i,:,:]*(1-mask[i,:,:])
                #k_complex2[i,:,:] = np.fft.fftshift(k_complex2[i,:,:])
                #k_complex2[i,:,:]=k_complex[i,:,:]
                NCSNpp_Image[i,:,:] = np.fft.ifft2(k_complex2[i,:,:])
                Image_NoDC[i, :, :] = np.fft.ifft2(k_complex[i, :, :])
            #NCSNpp_Image = NCSNpp_Image.transpose((2, 0, 1))



            # k_complex2[i,:,:] = np.fft.fftshift(k_complex2[i,:,:])
            mask_sos = np.sqrt(np.sum(np.square(np.abs(mask)),axis=0))
            mask_sos = mask_sos/np.max(np.abs(mask_sos))
            write_images(abs(mask_sos),osp.join(save_path,img_name+'Rec_mask'+'.png'))

            rec_kw_sos = np.sqrt(np.sum(np.square(np.abs(k_w)),axis=0))
            rec_kw_sos = rec_kw_sos/np.max(np.abs(rec_kw_sos))
            write_images(abs(rec_kw_sos),osp.join(save_path,img_name+'Rec_kw'+'.png'))

            rec_kdata_sos = np.sqrt(np.sum(np.square(np.abs(k_complex2)),axis=0))
            rec_kdata_sos = rec_kdata_sos/np.max(np.abs(rec_kdata_sos))
            write_images(abs(rec_kdata_sos),osp.join(save_path,img_name+'Rec_kdata'+'.png'))
            



            rec_Image_sos = np.sqrt(np.sum(np.square(np.abs(NCSNpp_Image)),axis=0))
            rec_Image_sos = rec_Image_sos/np.max(np.abs(rec_Image_sos))

            rec_Image_Nodc_sos = np.sqrt(np.sum(np.square(np.abs(Image_NoDC)), axis=0))
            rec_Image_Nodc_sos = rec_Image_Nodc_sos / np.max(np.abs(rec_Image_Nodc_sos))

            ori_data_sos = ori_data_sos[None,:]
            ori_data_sos = np.sqrt(np.sum(np.square(np.abs(ori_data_sos)), axis=0))
            ori_data_sos = ori_data_sos / np.max(np.abs(ori_data_sos))
            print(' rec_Image_sos_max:', np.max(rec_Image_sos), ' rec_Image_sos_min:', np.min(rec_Image_sos))
            print(' ori_data_sos_max:', np.max(ori_data_sos), ' ori_data_sos_min:', np.min(ori_data_sos))

            # Print PSNR
            psnr = compare_psnr(255*abs(rec_Image_sos),255*abs(ori_data_sos),data_range=255)
            ssim = compare_ssim(abs(rec_Image_sos),abs(ori_data_sos),data_range=1)
            print(' PSNR:', psnr,' SSIM:', ssim)

            psnr_nodc = compare_psnr(255 * abs(rec_Image_Nodc_sos), 255 * abs(ori_data_sos), data_range=255)
            ssim_nodc = compare_ssim(abs(rec_Image_Nodc_sos), abs(ori_data_sos), data_range=1)
            print(' psnr_nodc:', psnr_nodc, ' ssim_nodc:', ssim_nodc)

            write_images(abs(rec_Image_Nodc_sos), osp.join(save_path, img_name + 'Rec_noDC' + '.png'))
            write_images(abs(rec_Image_sos),osp.join(save_path,img_name+'Rec'+'.png'))
            io.savemat(osp.join(save_path,img_name+'rec_kdata.mat'),{'rec_kdata':k_complex2})  




            #sys.exit()

    except KeyboardInterrupt:
        LOGGER.warning('Interrupted by user')
    except Exception as ex:
        LOGGER.critical(f'Prediction failed due to {ex}:\n{traceback.format_exc()}')
        sys.exit(1)


if __name__ == '__main__':
    main()
