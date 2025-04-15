import os
from tqdm import tqdm

data_folder = '/home/b109/Desktop/XX/augmented_data_3channel_img (copy)'
files = os.listdir(data_folder)

# 遍历文件夹中的每个.mat文件并进行重命名
for i, file in enumerate(tqdm(files, desc="Renaming files")):
    if file.endswith('.mat'):
        src_filename = os.path.join(data_folder, file)
        dst_filename = os.path.join(data_folder, f'{i + 1}_a.mat')  # 使用编号重命名文件

        os.rename(src_filename, dst_filename)

data_folder = '/home/b109/Desktop/XX/augmented_data_3channel_img (copy)'
files = os.listdir(data_folder)

for i, file in enumerate(tqdm(files, desc="Renaming files")):
    if file.endswith('.mat'):
        src_filename = os.path.join(data_folder, file)
        dst_filename = os.path.join(data_folder, f'{i + 1}.mat')  # 使用编号重命名文件

        os.rename(src_filename, dst_filename)
print("文件已经按照阿拉伯数字重命名完成。")
