export TORCH_HOME=$(pwd) && export PYTHONPATH=$(pwd)


python3 bin/gen_mask_dataset_test.py \
$(pwd)/configs/data_gen/random_thick_256.yaml \
/home/b109/code/xx/DiffIR-master/DiffIR-master/DiffIR-inpainting/data/mri4000/val_256/val_source_256_test/ \
/home/b109/code/xx/DiffIR-master/DiffIR-master/DiffIR-inpainting/data/mri4000/val_256/random_thick_256_test/


CUDA_VISIBLE_DEVICES=0 python3 bin/predict.py \
model.path=$(pwd)/result/train/S2Epoch10/ \
indir=$(pwd)/data/mri4000/val_256/random_thick_256_test/ \
outdir=$(pwd)/inference/mri_random_thick_256_T_S2  model.checkpoint=8.ckpt

