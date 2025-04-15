export TORCH_HOME=$(pwd) && export PYTHONPATH=$(pwd)


# CUDA_VISIBLE_DEVICES=0 python3 bin/predict_coil.py \
# model.path=$(pwd)/result/train/S2_MaskLab_S1120_epoch100/ \
# indir=$(pwd)/data/mri4000/val_256/random_thick_256_test/ \
# outdir=$(pwd)/inference/mri_random_thick_256_coil_s2  model.checkpoint=51.ckpt   #51 38.15  0.948



# CUDA_VISIBLE_DEVICES=0 python3 bin/predict_coil.py \
# model.path=$(pwd)/result/train/S2_MaskLab_S1125_2_epoch100_sait_fast_light4/ \
# indir=$(pwd)/data/mri4000/val_256/random_thick_256_test/ \
# outdir=$(pwd)/inference/mri_random_thick_256_coil_s2  model.checkpoint=9.ckpt   # S2_MaskLab_fast_sait_epoch50_light4  20 38.7755 0.9576  w07   GEBRAIN
#                                                                                  # S2_MaskLab_S1132_epoch100_sait_fast_light4  last 38.35 0.9622 w06   GEBRAIN
#                                                                                 #S2_MaskLab_S1125_2_epoch100_sait_fast_light4  9 39.226  0.960  w06  GEBRAIN
#                                                                                 #S2_MaskLab_S1182_epoch100_sait_fast_light4   5   38.95   0.956  w06  GEBRAIN


# CUDA_VISIBLE_DEVICES=1 python3 bin/predict_coil.py \
# model.path=$(pwd)/result/train/S2_MaskLab_pei_S1162_1w2_light4_kw/ \
# indir=$(pwd)/data/mri4000/val_256/random_thick_256_test/ \
# outdir=$(pwd)/inference/mri_random_thick_256_coil_s2  model.checkpoint=18.ckpt   # S2_MaskLab_fast_sait_epoch50_light4  20 38.7755 0.9576  w07   GEBRAIN
                                                                                # S2_MaskLab_S1132_epoch100_sait_fast_light4  last 38.35 0.9622 w06   GEBRAIN
                                                                                #S2_MaskLab_S1125_2_epoch100_sait_fast_light4  9 39.226  0.960  w06  GEBRAIN
                                                                                #S2_MaskLab_S1182_epoch100_sait_fast_light4   5   38.95   0.956  w06  GEBRAIN
                                                                                #S2_MaskLab_S172_epoch100_fast_light4    41  38.9737   0.954  w07   GEBRAIN
                                                                                #S2_Masklab_S154_fast_sait_epoch1000_light4_head248816  64  39.20    0.9634 weight1_PeiBrain  GEBRAIN
                                                                                #S2_MaskLab_S172_9w_light4  6 39.1029  0.9609   GEBrain
                                                                                #S2_MaskLab_S1117_1w2_kw    1_4  38.288   0.959      4_3 38.820 0.9596
                                                                                #S2_MaskLab_S171_1w2_light4_innerVoid_kw   4_2   39.29   0.9578       5_1  38.65   0.96532  
                                                                                #S2_MaskLab_S1126_1w2_kw   3_0   39.12167 0.957
                                                                                #S2_MaskLab_pei_S1162_1w2_light4_kw   6_0  38.927  0.958
                                                                                #                                      12    39.32051  0.962
                                                                                #
                                                                                #

# CUDA_VISIBLE_DEVICES=0 python3 bin/predict_coil_mask.py \
# model.path=$(pwd)/result/train/S2_MaskLab_S1125_2_epoch100_sait_fast_light4/ \
# indir=$(pwd)/data/mri4000/val_256/random_thick_256_test/ \
# outdir=$(pwd)/inference/mri_random_thick_256_coil_s2  model.checkpoint=9.ckpt




# CUDA_VISIBLE_DEVICES=1 python3 bin/predict_coil_innerVoid.py \
# model.path=$(pwd)/result/train/S2_MaskLab_S171_1w2_light4_innerVoid_kw \
# indir=$(pwd)/data/mri4000/val_256/random_thick_256_test/ \
# outdir=$(pwd)/inference/mri_random_thick_256_coil_s2  model.checkpoint=4_2.ckpt         #S2_MaskLab_S171_1w2_light4_innerVoid_kw   4_2   39.29   0.9578       5_1  38.65   0.96532   


# CUDA_VISIBLE_DEVICES=0 python3 bin/predict_coil_innerVoid_innerAdd.py \
# --outdir $(pwd)/inference/mri_random_thick_256_coil_s2  \
# --path_void $(pwd)/result/train/S2_MaskLab_S125_fast_sait_light4_innerVoid_kw \
# --path_add $(pwd)/result/train/S2_MaskLab_S1125_2_epoch100_sait_fast_light4/ \
# --checkpoint_void last.ckpt \
# --checkpoint_add 9.ckpt 



# CUDA_VISIBLE_DEVICES=0 python3 bin/predict_coil_innerVoid_innerAdd.py \
# outdir=$(pwd)/inference/mri_random_thick_256_coil_s2_inner  \
# indir=$(pwd)/data/mri4000/val_256/random_thick_256_test/ \
# model.path_void=$(pwd)/result/train/S2_MaskLab_S125_fast_sait_light4_innerVoid_kw \
# model.path_add=$(pwd)/result/train/S2_MaskLab_S1125_2_epoch100_sait_fast_light4/ \
# model.checkpoint_void=last.ckpt \
# model.checkpoint_add=9.ckpt 



# CUDA_VISIBLE_DEVICES=0 python3 bin/predict_coil_innerVoid_innerAdd.py \
# outdir=$(pwd)/inference/mri_random_thick_256_coil_s2_inner  \
# indir=$(pwd)/data/mri4000/val_256/random_thick_256_test/ \
# model.path_void=$(pwd)/result/train/S2_MaskLab_S171_1w2_light4_innerVoid_kw \
# model.path_add=$(pwd)/result/train/S2_MaskLab_S1125_2_epoch100_sait_fast_light4/ \
# model.checkpoint_void=4_2.ckpt \
# model.checkpoint_add=9.ckpt




# CUDA_VISIBLE_DEVICES=0 python3 bin/predict_coil_innerVoid_innerAdd_average.py \
# outdir=$(pwd)/inference/mri_random_thick_256_coil_s2_inner  \
# indir=$(pwd)/data/mri4000/val_256/random_thick_256_test/ \
# model.path_void=$(pwd)/result/train/S2_MaskLab_S171_1w2_light4_innerVoid_kw \
# model.path_add=$(pwd)/experiments/inpainting_2024-06-30_10-20-18_train_DiffIRS2-mri_/ \
# model.checkpoint_void=4_2.ckpt \
# model.checkpoint_add=last.ckpt    

# 


#多step单个
# CUDA_VISIBLE_DEVICES=0 python3 bin/predict_coil.py \
# outdir=$(pwd)/result/test  \
# model.path=$(pwd)/experiments/inpainting_2024-08-07_14-29-08_train_DiffIRS2-mri_ \
# indir=$(pwd)/data/mri4000/val_256/random_thick_256_test/ \
# model.checkpoint=last.ckpt


#step1 : PSNR: 14.130041247568304  SSIM: 0.3621511588287542  mse 386.3633075155175
#step2 :  PSNR: 32.95850293031356  SSIM: 0.8768038286705464  mse 5.059990561489781
#step4 :  PSNR: 35.35497413453726  SSIM: 0.8961716888344288  mse 2.9140874793506923
#step8 :  PSNR: 33.452184058355535  SSIM: 0.9009259499732554  mse 4.516287630823717
#step16 : PSNR: 33.516190542924356  SSIM: 0.906738047076404  mse 4.450214552962131
#step32 :  PSNR: 36.22051332882474  SSIM: 0.9098847910299301  mse 2.3875290712226467





# #单个
# CUDA_VISIBLE_DEVICES=1 python3 bin/predict_coil.py \
# outdir=$(pwd)/result/test1  \
# model.path=$(pwd)/result/train/S2_MaskLab_pei_S1162_1w2_light4_kw/ \
# indir=$(pwd)/data/mri4000/val_256/random_thick_256_test/ \
# model.checkpoint=12.ckpt

#多个
CUDA_VISIBLE_DEVICES=1 python3 bin/predict_coil_innerVoid_innerAdd_average_1coil.py \
outdir=$(pwd)/result/test \
indir=$(pwd)/data/mri4000/val_256/random_thick_256_test/ \
model.path_void=$(pwd)/result/train/S2_MaskLab_S171_1w2_light4_innerVoid_kw \
model.path_add=$(pwd)/result/train/S2_MaskLab_S1126_1w2_kw/ \
model.checkpoint_void=4_2.ckpt \
model.checkpoint_add=44.ckpt                           


#模型文件夹   S2_MaskLab_S171_1w2_light4_innerVoid_kw  +   S2_MaskLab_S1125_2_epoch100_sait_fast_light4
#模型编号                   4_2                                 9
#结果               PSNR: 39.778160587748175               SSIM: 0.9687089932601451
#                   
#                          5_1                                 9
#                   PSNR: 39.103077438636554                SSIM: 0.9704380905022622            


#              S2_MaskLab_S171_1w2_light4_innerVoid_kw          S2_MaskLab_S1117_1w2_kw     
#                           4_2                                    4_3
#               PSNR: 39.57684139552248                       SSIM: 0.966562115490675
#
#                           4_2                                     14
#               PSNR: 39.58039977052825                         SSIM: 0.9650424863784314


#               S2_MaskLab_S171_1w2_light4_innerVoid_kw      S2_MaskLab_S1126_1w2_kw
#                       4_2                                         44                      weight_GE_brain_09
#                  PSNR: 39.871985086350925  SSIM: 0.9676736292259693              
#                       4_2                                         44_1
#                 PSNR: 39.79798177365948  SSIM: 0.9664248849174977

#
#               S2_MaskLab_S171_1w2_light4_innerVoid_kw             S2_MaskLab_pei_S1162_1w2_light4_kw
#                       4_2                                                     12
#                PSNR: 39.857919636928024                           SSIM: 0.9673917989370823
#                       4_2                                                      38
#                PSNR: 39.89776469976391                            SSIM: 0.9672417993795872
#                       4_2                                                     45
#                PSNR: 39.783117684562086                           SSIM: 0.9665205175700188
