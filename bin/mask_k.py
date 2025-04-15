import os
import random
import numpy as np
import scipy.io as sio

def multiply_random_mask_k_space(input_array):
    #folder_path = "/home/b109/code/xx/DiffIR-master/DiffIR-master/DiffIR-inpainting/data/mask"  
    folder_path = "/home/b109/Desktop/XX/inpainting/data/mask"  

    # 获取所有子文件夹路径
    subfolders = [subfolder for subfolder in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, subfolder))]

    if len(subfolders) == 0:
        raise ValueError("文件夹中没有子文件夹")

    # 随机选择一个子文件夹
    random_subfolder = random.choice(subfolders)

    # 子文件夹路径
    subfolder_path = os.path.join(folder_path, random_subfolder)

    # 获取子文件夹中所有的mask.mat文件路径
    mask_files = [file for file in os.listdir(subfolder_path) if file.endswith(".mat")]

    if len(mask_files) == 0:
        raise ValueError("子文件夹中没有mask.mat文件")

    random_mask_file = random.choice(mask_files)

    # 读取选中的mask.mat文件
    mask_data = sio.loadmat(os.path.join(subfolder_path, random_mask_file))
    # print("选取的mask路径：",os.path.join(subfolder_path, random_mask_file))
    

    mask = None
    for key in mask_data.keys():
        if isinstance(mask_data[key], np.ndarray) and mask_data[key].shape == input_array.shape:
            mask = mask_data[key]
            break

    if mask is None:
        raise ValueError("找不到合适的mask")

    # 将输入数据和mask相乘
    #result = np.multiply(input_array, mask)
    # print("MASK.SHAPE=",mask.shape)
    return mask