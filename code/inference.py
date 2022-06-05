import os
import torch
import numpy as np
import torchvision.models as models
from PIL import Image
import torch.nn as nn
import torchvision.transforms as tf
from utils import seed_everything
from data import SmokeDataset_test
from torch.utils.data import DataLoader

def inference(img_list):
    seed_everything(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(in_features=2048, out_features=2, bias=True)
    model.load_state_dict(torch.load('../pt/pytorch.bin'))
    model.to(device)
    model.eval()
    all_prediction=[]
    ds = SmokeDataset_test(img_list)
    dl = DataLoader(ds, batch_size=8, shuffle=False)
    for image in dl:
        output = model(image)
        _, pred = torch.max(output,1)
        pred = pred.cpu().numpy()
        all_prediction.extend(pred)
    return all_prediction