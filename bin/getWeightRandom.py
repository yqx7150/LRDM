import os
import glob
import random
import scipy.io

def read_random_mat(folder_path):
    # 找到文件夹中所有的.mat文件
    mat_files = glob.glob(os.path.join(folder_path, '*.mat'))

    # 从文件列表中随机选择一个.mat文件
    random_mat = random.choice(mat_files)

    # 读取选定的.mat文件内容
    mat_data = scipy.io.loadmat(random_mat)['weight']

    return mat_data



