import numpy as np
import torchvision.datasets
import torchvision.models as models
import torch.utils.data as data
from torchvision import transforms
import os ,torch
import torch.nn as nn
from torch.utils.data import random_split


class Res18Feature(nn.Module):
    def __init__(self, pretrained=True, num_classes=8, drop_rate=0):
        super(Res18Feature, self).__init__()
        self.drop_rate = drop_rate
        resnet = models.resnet18(pretrained)
        # self.feature = nn.Sequential(*list(resnet.children())[:-2]) # before avgpool
        self.features = nn.Sequential(*list(resnet.children())[:-1])  # after avgpool 512x1

        fc_in_dim = list(resnet.children())[-1].in_features  # original fc layer's in dimention 512

        self.fc = nn.Linear(fc_in_dim, num_classes)  # new fc layer 512x7
        self.alpha = nn.Sequential(nn.Linear(fc_in_dim, 1), nn.Sigmoid())

    def forward(self, x):
        x = self.features(x)

        if self.drop_rate > 0:
            x = nn.Dropout(self.drop_rate)(x)
        x = x.view(x.size(0), -1)

        attention_weights = self.alpha(x)
        out = attention_weights * self.fc(x)
        return attention_weights, out
model_save_path = "/home/daip/share/HDD-6TB/project/EmoProject/deep_model/emo.pth"

def retrain(data_path,save_path):
    save_model = os.path.join(save_path,'weight')
    if not os.path.exists(save_model):
        os.makedirs(save_model)
    res18 = Res18Feature(pretrained=False)
    res18 = torch.load(model_save_path)
    res18.cuda()
    # res18.train()
    data_transforms = transforms.Compose([
        # transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])
    full_data = torchvision.datasets.ImageFolder(root=data_path, transform=data_transforms)
    print(full_data.class_to_idx)
    train_size = int(len(full_data) * 0.7)  # 这里train_size是一个长度矢量，并非是比例，我们将训练和测试进行8/2划分
    test_size = len(full_data) - train_size
    train_dataset, val_dataset = random_split(full_data, [train_size, test_size])
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=256,
                                               num_workers=0,
                                               shuffle=True,
                                               pin_memory=True)
    print('Train set size:', train_dataset.__len__())
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=256,
                                             num_workers=0,
                                             shuffle=False,
                                             pin_memory=True)
    print('Validation set size:', val_dataset.__len__())

    params = res18.parameters()
    optimizer = torch.optim.SGD(params, lr=0.01,
                                momentum=0.9,
                                weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.9)
    res18 = res18.cuda()
    criterion = torch.nn.CrossEntropyLoss()

    margin_1 = 0.15
    beta = 0.7
    for i in range(1, 1 + 1):
        running_loss = 0.0
        correct_sum = 0
        iter_cnt = 0
        res18.train()
        for batch_i, (imgs, targets) in enumerate(train_loader):
            batch_sz = imgs.size(0)
            iter_cnt += 1
            tops = int(batch_sz * beta)
            optimizer.zero_grad()
            imgs = imgs.cuda()
            attention_weights, outputs = res18(imgs)

            # Rank Regularization
            _, top_idx = torch.topk(attention_weights.squeeze(), tops)
            _, down_idx = torch.topk(attention_weights.squeeze(), batch_sz - tops, largest=False)

            high_group = attention_weights[top_idx]
            low_group = attention_weights[down_idx]
            high_mean = torch.mean(high_group)
            low_mean = torch.mean(low_group)
            # diff  = margin_1 - (high_mean - low_mean)
            diff = low_mean - high_mean + margin_1

            if diff > 0:
                RR_loss = diff
            else:
                RR_loss = 0.0

            targets = targets.cuda()
            loss = criterion(outputs, targets) + RR_loss
            loss.backward()
            optimizer.step()

            running_loss += loss
            _, predicts = torch.max(outputs, 1)
            correct_num = torch.eq(predicts, targets).sum()
            correct_sum += correct_num
        scheduler.step()
        acc = correct_sum.float() / float(train_dataset.__len__())
        running_loss = running_loss / iter_cnt
        print('[Epoch %d] Training accuracy: %.4f. Loss: %.3f' % (i, acc, running_loss))

        with torch.no_grad():
            running_loss = 0.0
            iter_cnt = 0
            bingo_cnt = 0
            sample_cnt = 0
            res18.eval()
            for batch_i, (imgs, targets) in enumerate(val_loader):
                _, outputs = res18(imgs.cuda())
                targets = targets.cuda()
                loss = criterion(outputs, targets)
                running_loss += loss
                iter_cnt += 1
                _, predicts = torch.max(outputs, 1)
                correct_num = torch.eq(predicts, targets)
                bingo_cnt += correct_num.sum().cpu()
                sample_cnt += outputs.size(0)

            running_loss = running_loss / iter_cnt
            acc = bingo_cnt.float() / float(sample_cnt)
            acc = np.around(acc.numpy(), 4)
            print("[Epoch %d] Validation accuracy:%.4f. Loss:%.3f" % (i, acc, running_loss))
    torch.save(res18,os.path.join(save_model, "emo"+".pth"))
    # torch.save(res18,'/home/daip/share/HDD-6TB/project/EmoProject/deep_model/emo.pth')

# retrain('/home/daip/share/HDD-6TB/DAIP-Datasets/emotion/test/','/home/daip/share/old_share/wxf/Self-Cure-Network-master/emo_project/models_5_7')
