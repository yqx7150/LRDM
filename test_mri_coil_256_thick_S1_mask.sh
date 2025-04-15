export TORCH_HOME=$(pwd) && export PYTHONPATH=$(pwd)


# # 原模型
# CUDA_VISIBLE_DEVICES=1 python3 bin/predict_coil.py \
# model.path=$(pwd)/result/train/masklab_epoch400/ \
# indir=$(pwd)/data/mri4000/val_256/random_thick_256_test/ \
# outdir=$(pwd)/inference/mri_random_thick_256_coil model.checkpoint=120.ckpt

#四层模型
CUDA_VISIBLE_DEVICES=0 python3 bin/predict_coil.py \
model.path=$(pwd)/result/train/MaskLab_Pei_1w2_kw_light/ \
indir=$(pwd)/data/mri/val_256/random_thick_256_test/ \
outdir=$(pwd)/inference/mri_random_thick_256_coil model.checkpoint=162.ckpt    #epoch400:120     #masklab_epoch400_light4  130  36.82  0.923
                                                                                                #masklab_fast_siat_epoch50_light4    50     37.88116924084968   0.95269881148386
                                                                                                #masklab_epoch1000_light_temp   182   38.74
                                                                                                #masklab_fast_epoch400_temp 72 37.442
                                                                                                #masklab_fast_epoch400_light4_head12488_temp  54  36.975   0.929 
                                                                                                #masklab_fast_sait_epoch400_light4_head13138816_temp   47  36.51   0.950
                                                                                                #masklab_fast_sait_epoch400_light4_head248816_temp   54  38.629  0.949
                                                                                                #MaskLab_9W_light4  72 38.22   0.9560
                                                                                                #MaskLab_1w2_kw_light4_33epoch   35   38.23   0.950
                                                                                                #MaskLab_1w2_kw_light4_61+  117   39.238635  0.9578
                                                                                                #                           126   38.92944 0.95569
                                                                                                #MaskLab_Pei_1w2_kw_light   81   38.9788    0.9562
                                                                                                #                           156  38.95   0.956
                                                                                                #                           162  39.137 0.9611

# CUDA_VISIBLE_DEVICES=0 python3 bin/predict_coil.py \
# model.path=$(pwd)/result/train/epoch40/ \
# indir=$(pwd)/data/mri4000/val_256/random_thick_256_test/ \
# outdir=$(pwd)/inference/mri_random_thick_256_coil model.checkpoint=last.ckpt   

#中间挖掉
# CUDA_VISIBLE_DEVICES=1 python3 bin/predict_coil_innerVoid.py \
# model.path=$(pwd)/result/train/MaskLab_1w2_innerVoid_light4/ \
# indir=$(pwd)/data/mri4000/val_256/random_thick_256_test/ \
# outdir=$(pwd)/inference/mri_random_thick_256_coil model.checkpoint=71.ckpt    #masklab_fast_sait_light4_innerVoid_kw    25
                                                                              #MaskLab_1w2_innerVoid_light4_37epoch     37   38.27
                                                                              #MaskLab_1w2_innerVoid_light4    24   38.91   0.9529


