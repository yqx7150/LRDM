export TORCH_HOME=$(pwd) && export PYTHONPATH=$(pwd)


# python3 bin/gen_mask_dataset.py \
# $(pwd)/configs/data_gen/random_thick_256.yaml \
# /home/b109/code/xx/DiffIR-master/DiffIR-master/DiffIR-inpainting/data/celeba-hq-dataset/val_source_256/ \
# /home/b109/code/xx/DiffIR-master/DiffIR-master/DiffIR-inpainting/data/celeba-hq-dataset/val_256/random_thick_256/


python3 bin/gen_mask_dataset_test.py \
$(pwd)/configs/data_gen/random_thick_256.yaml \
/home/b109/Desktop/XX/inpainting/data/mri/val_256/val_source_256_test/ \
/home/b109/Desktop/XX/inpainting/data/mri/val_256/random_thick_256_test/



# python3 bin/gen_mask_dataset.py \
# $(pwd)/configs/data_gen/random_thin_256.yaml \
# /mnt/bn/xiabinpaint/dataset/celeba-hq-dataset/val_source_256/ \
# /mnt/bn/xiabinpaint/dataset/celeba-hq-dataset/val_256/random_thin_256/

# python3 bin/gen_mask_dataset.py \
# $(pwd)/configs/data_gen/random_medium_256.yaml \
# /mnt/bn/xiabinpaint/dataset/celeba-hq-dataset/val_source_256/ \
# /mnt/bn/xiabinpaint/dataset/celeba-hq-dataset/val_256/random_medium_256/

# python3 bin/gen_mask_dataset.py \
# $(pwd)/configs/data_gen/random_thick_256.yaml \
# /mnt/bn/xiabinpaint/dataset/celeba-hq-dataset/visual_test_source_256/ \
# /mnt/bn/xiabinpaint/dataset/celeba-hq-dataset/visual_test_256/random_thick_256/

# python3 bin/gen_mask_dataset.py \
# $(pwd)/configs/data_gen/random_thin_256.yaml \
# /mnt/bn/xiabinpaint/dataset/celeba-hq-dataset/visual_test_source_256/ \
# /mnt/bn/xiabinpaint/dataset/celeba-hq-dataset/visual_test_256/random_thin_256/

# python3 bin/gen_mask_dataset.py \
# $(pwd)/configs/data_gen/random_medium_256.yaml \
# /mnt/bn/xiabinpaint/dataset/celeba-hq-dataset/visual_test_source_256/ \
# /mnt/bn/xiabinpaint/dataset/celeba-hq-dataset/visual_test_256/random_medium_256/
