import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.transforms import transforms

from model.resnet import ResNet18
from PIL import Image
import cv2
import numpy as np
import os
import torchvision.transforms as transforms


image_name ="camel-1536x580.jpg"
idx = 354

image_name = 'handchet.jpg'
idx = 596
image_name ="handkerchief.jpg"
idx = 591

image_name ="frying_pan.jpg"
idx = 567

image_name ="cup.jpg"
idx = 968
image_name ="wallet.jpg"
idx = 893

device = 'cuda' if torch.cuda.is_available() else 'cpu'

img = Image.open("./data_file/{}".format(image_name)).convert('RGB')
segmented_img = np.array(Image.open("./data_file/{}".format(image_name)).convert('RGB'), dtype=np.uint8)

transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
         ])
img = transform(img)
img=img.unsqueeze(dim=0).to(device)


model = ResNet18(pretrained=True)  # load model
model.to(device) # load to GPU or CPU
model.eval()

response_map = model(img)
response_map = F.interpolate(response_map, size=(img.shape[2], img.shape[3]), mode='bilinear')
response_map = F.softmax(response_map, dim=1).squeeze()
response_map = response_map.data.cpu().numpy()

mask = response_map[idx, :, :] < 0.5
response_map = response_map[idx, :, :] * 255.0
segmented_img[mask, :] = 0
segmented_img = segmented_img[:, :, ::-1]
import pdb;pdb.set_trace()
cv2.imwrite('./result/{}_response.jpg'.format(image_name.split(".")[0]), response_map)
cv2.imwrite('./result/{}_mask.jpg'.format(image_name.split(".")[0]), segmented_img)
#cv2.imwrite('./{}_total.jpg'.format(imagme_name), )