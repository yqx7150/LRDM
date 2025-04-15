import os
from scipy.io import loadmat
from tqdm import tqdm

folder_path = '/home/b109/Desktop/XX/augmented_data_3channel_img (copy)'

# 获取删除前文件总数
before_delete_count = len(os.listdir(folder_path))

for file_name in tqdm(os.listdir(folder_path), desc="Processing files"):
    file_path = os.path.join(folder_path, file_name) 
    try:
        mat_content = loadmat(file_path)
        if not mat_content:
            print(f"Empty file: {file_path}")
            os.remove(file_path)
    except Exception as e:
        print(f"Unable to open file: {file_path}")
        os.remove(file_path)

# 获取删除后文件总数
after_delete_count = len(os.listdir(folder_path))

print(f"Total files before delete: {before_delete_count}")
print(f"Total files after delete: {after_delete_count}")