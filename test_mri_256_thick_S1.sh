export TORCH_HOME=$(pwd) && export PYTHONPATH=$(pwd)


# CUDA_VISIBLE_DEVICES=0 python3 bin/predict.py \
# model.path=$(pwd)/result/train/inpainting_2024-04-10_17-42-51_train_DiffIRS1-mri_/ \
# indir=$(pwd)/data/mri4000/val_256/random_thick_256_test/ \
# outdir=$(pwd)/inference/mri_random_thick_256_T model.checkpoint=last.ckpt


CUDA_VISIBLE_DEVICES=0 python3 bin/predict.py \
model.path=$(pwd)/result/train/epoch15/ \
indir=$(pwd)/data/mri4000/val_256/random_thick_256_test/ \
outdir=$(pwd)/inference/mri_random_thick_256_T model.checkpoint=last.ckpt

