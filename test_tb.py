from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image
write = SummaryWriter("logs")
image_path = "data/train/bees_image/16838648_415acd9e3f.jpg"
img_PIL = Image.open(image_path)
img_array = np.array(img_PIL)

write.add_image("test", img_array, 2, dataformats='HWC')
for i in range(100):
    write.add_scalar("y=2x", 2*i, i)
write.close()