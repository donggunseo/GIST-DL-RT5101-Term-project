import os
import torch
from sklearn.model_selection import train_test_split
import numpy as np
import random
from data import SmokeDataset
from torch.utils.data import DataLoader
import torchvision.models as models
import torch.nn as nn
from sklearn.metrics import f1_score

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 32
LR = 1e-3
EPOCHS = 100
early_stop = 8
nonsmoking_dir = '../SmokeImage/nonsmoking'
smoking_dir = '../SmokeImage/smoking'
nonsmoking_image_list = os.listdir(nonsmoking_dir)
smoking_image_list = os.listdir(smoking_dir)
smoking_image_list = [os.path.join(smoking_dir, x) for x in smoking_image_list]
nonsmoking_image_list = [os.path.join(nonsmoking_dir, x) for x in nonsmoking_image_list]
img_list = smoking_image_list + nonsmoking_image_list
label_list = [0]*len(smoking_image_list) + [1]*len(nonsmoking_image_list)

x_train, x_valid, y_train, y_valid = train_test_split(img_list, label_list, test_size = 0.1, shuffle=True, stratify=label_list, random_state=42)

train_dataset = SmokeDataset(x_train, y_train)
valid_dataset = SmokeDataset(x_valid, y_valid, False)

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

model = models.resnet50(pretrained=False).to(device)
model.fc = nn.Linear(in_features=2048, out_features=2, bias=True)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr = LR)
# lr_scheduler = torch.optim.

best_f1 = 0
early_stop_count = 0
for epoch in range(EPOCHS):
    print('*** Epoch {} ***'.format(epoch))
    model.train()
    total_train_loss = []
    total_train_acc = []
    total_valid_loss = []
    total_valid_acc = []
    total_valid_f1 = []
    for x,y in train_dataloader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        total_train_loss.append(loss.cpu().item())
        _, pred = torch.max(output,1)
        total_train_acc.extend((pred==y).cpu().tolist())
        print('loss : ', loss.cpu().item())
    model.eval()
    for x,y in valid_dataloader:
        x = x.to(device)
        y = y.to(device)
        output = model(x)
        loss = criterion(output, y)
        _, pred = torch.max(output,1)
        total_valid_loss.append(loss.cpu().item())
        total_train_acc.extend((pred==y).cpu().tolist())
        total_valid_f1.append(f1_score(y_true = y.cpu().numpy(), y_pred = pred.cpu().numpy(), average='macro'))
    
    epoch_train_loss = np.mean(total_train_loss)
    epoch_valid_loss = np.mean(total_valid_loss)
    epoch_train_acc = np.mean(total_train_acc) * 100
    epoch_valid_acc = np.mean(total_valid_acc) * 100
    epoch_f1 = np.mean(total_valid_f1)
    print(
                    f"[Epoch {epoch}] \n"
                    f"train loss : {epoch_train_loss:.4f} | train acc : {epoch_train_acc:.2f}% \n"
                    f"valid loss : {epoch_valid_loss:.4f} | valid acc : {epoch_valid_acc:.2f}% | valid f1 score : {epoch_f1:.4f}"
    )
    if epoch_f1> best_f1:
        best_f1 = epoch_f1
        early_stop_count = 0
        print('update best model')
        torch.save(model.state_dict(), './pytorch.bin')
    else:
        early_stop_count+=1
    if early_stop_count == early_stop:
        print("early stopped")
        break



        








