import torch
path = "/home/b109/Desktop/XX/inpainting/result/train/MaskLab_Pei_1w2_kw_light/models/162.ckpt"


save_path = "/home/b109/Desktop/XX/inpainting/result/train/S1forS2/mri-S1-masklab_pei_S1162_1w2_light4_kw.pth"
#path = "./experiments/DiffIRS1-place/models/last.ckpt"
#save_path = "./place-S1.pth"
#path = "./experiments/Big-DiffIRS1-place/models/last.ckpt"
#save_path = "./placebigdata-S1.pth"
s=torch.load(path)
# for k,v in s.items():
#    print(k)

# for k,v in s["state_dict"].items():
#     if "evaluator" not in k and "loss" not in k and "discriminator" not in k:
#         print(k)
# print(s["state_dict"])
new={}
for k,v in s["state_dict"].items():
    if "evaluator" not in k and "loss" not in k and "discriminator" not in k:
        k=k[10:]
        print(k)
        new[k]=v

for k,v in new.items():
    print(k)
torch.save(new,save_path)
