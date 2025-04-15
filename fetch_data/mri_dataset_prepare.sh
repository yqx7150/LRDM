# mkdir celeba-hq-dataset

# unzip data256x256.zip -d celeba-hq-dataset/

# Reindex
# for i in `echo {00001..010000}`
# do
#     #mv '/home/b109/code/xx/DiffIR-master/DiffIR-master/DiffIR-inpainting/data/celeba-hq-dataset/data256x256/'$i'.jpg' '/home/b109/code/xx/DiffIR-master/DiffIR-master/DiffIR-inpainting/data/celeba-hq-dataset/data256x256/'$[10#$i - 1]'.jpg'
#     mv '/home/b109/code/xx/DiffIR-master/DiffIR-master/DiffIR-inpainting/data/celeba-hq-dataset/img_align_celeba/'$i'.jpg' '/home/b109/code/xx/DiffIR-master/DiffIR-master/DiffIR-inpainting/data/celeba-hq-dataset/img_align_celeba/'$[10#$i - 1]'.jpg'
# done


# Split: split train -> train & val
# cat fetch_data/train_shuffled.flist | shuf > /home/b109/code/xx/DiffIR-master/DiffIR-master/DiffIR-inpainting/data/mri4000/temp_train_shuffled.flist
# cat /home/b109/code/xx/DiffIR-master/DiffIR-master/DiffIR-inpainting/data/mri4000/temp_train_shuffled.flist | head -n 2000 > /home/b109/code/xx/DiffIR-master/DiffIR-master/DiffIR-inpainting/data/mri4000/val_shuffled.flist
# #cat /home/b109/code/xx/DiffIR-master/DiffIR-master/DiffIR-inpainting/data/mri4000/temp_train_shuffled.flist | tail -n +2001 >/home/b109/code/xx/DiffIR-master/DiffIR-master/DiffIR-inpainting/data/mri4000/train_shuffled.flist
# cat fetch_data/val_shuffled.flist > /home/b109/code/xx/DiffIR-master/DiffIR-master/DiffIR-inpainting/data/celeba-hq-dataset/visual_test_shuffled.flist

mkdir /home/b109/code/xx/DiffIR-master/DiffIR-master/DiffIR-inpainting/data/mri4000/train_256/
mkdir /home/b109/code/xx/DiffIR-master/DiffIR-master/DiffIR-inpainting/data/mri4000/val_source_256/
mkdir /home/b109/code/xx/DiffIR-master/DiffIR-master/DiffIR-inpainting/data/mri4000/visual_test_source_256/

# cat /home/b109/code/xx/DiffIR-master/DiffIR-master/DiffIR-inpainting/data/celeba-hq-dataset/train_shuffled.flist | xargs -I {} mv /home/b109/code/xx/DiffIR-master/DiffIR-master/DiffIR-inpainting/data/celeba-hq-dataset/img_align_celeba/{} /home/b109/code/xx/DiffIR-master/DiffIR-master/DiffIR-inpainting/data/celeba-hq-dataset/train_256/
# cat /home/b109/code/xx/DiffIR-master/DiffIR-master/DiffIR-inpainting/data/celeba-hq-dataset/val_shuffled.flist | xargs -I {} mv /home/b109/code/xx/DiffIR-master/DiffIR-master/DiffIR-inpainting/data/celeba-hq-dataset/img_align_celeba/{} /home/b109/code/xx/DiffIR-master/DiffIR-master/DiffIR-inpainting/data/celeba-hq-dataset/val_source_256/
# cat /home/b109/code/xx/DiffIR-master/DiffIR-master/DiffIR-inpainting/data/celeba-hq-dataset/visual_test_shuffled.flist | xargs -I {} mv /home/b109/code/xx/DiffIR-master/DiffIR-master/DiffIR-inpainting/data/celeba-hq-dataset/img_align_celeba/{} /home/b109/code/xx/DiffIR-master/DiffIR-master/DiffIR-inpainting/data/celeba-hq-dataset/visual_test_source_256/


# create location config celeba.yaml
PWD=$(pwd)
DATASET=/home/b109/code/xx/DiffIR-master/DiffIR-master/DiffIR-inpainting/data/mri4000
MRI=${PWD}/configs/training/location/mri.yaml

touch $MRI
echo "# @package _group_" >> $MRI
echo "data_root_dir: ${DATASET}/" >> $MRI
echo "out_root_dir: ${PWD}/experiments/" >> $MRI
echo "tb_dir: ${PWD}/tb_logs/" >> $MRI
echo "pretrained_models: ${PWD}/" >> $MRI
