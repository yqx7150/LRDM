import os

folder_path = '/home/b109/Desktop/XX/augmented_data_3channel_img (copy)'

for file_name in os.listdir(folder_path):
    file_path = os.path.join(folder_path, file_name) 
    try:
        with open(file_path, 'r') as file:
            file_content = file.read()
            if not file_content.strip():
                print(f"Empty file: {file_path}")
    except Exception as e:
        print(f"Unable to open file: {file_path}")