from PIL import Image
from torchvision import transforms
import torch
import torch.nn as nn
import torchvision.utils as vutils
import os

from model.PAGRN_recurrent import PAGRN

import numpy as np
import time
from tools.load_pretrain_model import load_pretrain_model
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "7"

def main():
    img = Image.open('img.png').convert('RGB')
    transform = transforms.Compose([
        transforms.Resize([400, 400]),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    input = transform(img)
    input = input.unsqueeze(0)

    model = PAGRN().cuda()
    pth_path = 
    model.load_state_dict(torch.load(pth_path))
    output1, output = model(input.cuda())
    # output = output.squeeze().detach().numpy()
    vutils.save_image(output[0], 'test_result.png', normalize=False)
    print("end test")

if __name__ == '__main__':
    main()