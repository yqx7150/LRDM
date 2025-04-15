import os
import random
import shutil
from tqdm import tqdm

import sys

# 源文件夹路径和目标文件夹路径
source_folder = '/home/b109/Desktop/XX/fastMRI_img_IVI_mat_1w2 (copy)'
target_folder = '/home/b109/Desktop/XX/inpainting/data/mri/val_256/val_source_256_test'

# 遍历源文件夹中的文件
files = os.listdir(source_folder)
total_files = len(files)
print(total_files)


# 每隔1000个文件进行一次选择
for i in tqdm(range(0, total_files, 1000)):
    # 随机选择一个文件
    random_file = random.choice(files[i:i+1000])
    
    # 构建源文件和目标文件的路径
    source_path = os.path.join(source_folder, random_file)
    target_path = os.path.join(target_folder, random_file)
    
    # 将文件剪切到目标文件夹中
    shutil.move(source_path, target_path)
    tqdm.write(f'Moved file {random_file} to target folder')
