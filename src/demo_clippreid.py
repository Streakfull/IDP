import pdb
from clipreid.config import imagedata_kwargs, get_default_config
import torchreid
from torchreid.data import ImageDataManager
import os
import torch
import torchvision.transforms as transforms
from PIL import Image

cfg = get_default_config()
cfg = imagedata_kwargs(cfg)
datamanager = torchreid.data.ImageDataManager(**cfg)
train_loader = datamanager.train_loader
val_loader = datamanager.test_loader['soccernetv3']['query']


enumartiont = enumerate(train_loader)

save_dir = "../logs/vis/reid"
os.makedirs(save_dir, exist_ok=True)
for i, batch in enumartiont:

    # for j in range(batch['img'].size(0)):  # Iterate through batch
    #     img_tensor = batch['img'][j]  # Get image tensor
    #     pid = batch['pid'][j].item()  # Convert tensor to int

    #     # Convert tensor to PIL Image
    #     img = transforms.ToPILImage()(img_tensor)

    #     # Define image save path
    #     save_path = os.path.join(save_dir, f"{pid}_{j}.jpg")

    #     # Save the image
    #     img.save(save_path)

    #     print(f"Saved {save_path}")
    import pdb
    pdb.set_trace()
