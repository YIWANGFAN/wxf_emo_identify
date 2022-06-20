# -*- coding: utf-8 -*-
# @Time    : 2022/4/26 16:33
# @Author  : wxf
# @FileName: Retrain.py
# @Software: PyCharm
# @Email ：15735952634@163.com
from io import BytesIO
from flask import Flask,request
from flask_cors import CORS
from PIL import Image
from torchvision import transforms
from torch.autograd import Variable
import os
import random
import  torch
from emo_retrain import retrain


app = Flask(__name__)
cors = CORS(app, resources={r"*": {"origins": "*"}})

import torchvision.models as models
import torch.nn as nn
class Res18Feature(nn.Module):
    def __init__(self, pretrained, num_classes=7):
        super(Res18Feature, self).__init__()
        resnet = models.resnet18(pretrained)
        # self.feature = nn.Sequential(*list(resnet.children())[:-2]) # before avgpool
        self.features = nn.Sequential(*list(resnet.children())[:-1])  # after avgpool 512x1

        fc_in_dim = list(resnet.children())[-1].in_features  # original fc layer's in dimention 512

        self.fc = nn.Linear(fc_in_dim, num_classes)  # new fc layer 512x7
        self.alpha = nn.Sequential(nn.Linear(fc_in_dim, 1), nn.Sigmoid())

    def forward(self, x):
        x = self.features(x)

        x = x.view(x.size(0), -1)

        attention_weights = self.alpha(x)
        out = attention_weights * self.fc(x)
        return attention_weights, out


# 模型存储路径
model_save_path = "/home/daip/share/HDD-6TB/project/EmoProject/deep_model/emo.pth"  # 修改为你自己保存下来的模型文件
# img_path = "/home/daip/share/HDD-6TB/DAIP-Datasets/emotion/test/anger/image0001375.jpg"  # 待测试照片位置

# ------------------------ 加载数据 --------------------------- #

preprocess_transform = transforms.Compose([
    # transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

# res18 = Res18Feature(pretrained=False)
# res18 = torch.load(model_save_path)
# checkpoint = torch.load(model_save_path)
# res18.load_state_dict(checkpoint['model_state_dict'])

emo_list = ['anger', 'contempt', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

@app.route('/retrain',methods=['POST'])
def get_1():
    try:
        get_data = request.form.to_dict()
        print(get_data)
        data_root = get_data.get('data_root')
        save_path = get_data.get('save_path')
        print(data_root)
        print(save_path)

        retrain(data_root,save_path)
        print('#' * 100)
        result = 'ok'
        return result
    except:
        result = 'false'
        return result


@app.route('/predict', methods=['POST'])
def predict():
    res18 = torch.load(model_save_path)
    res18.cuda()
    res18.eval()
    get_image = request.files.get('img').read()
    get_data = request.form.to_dict()
    x = int(float(get_data.get('x')))
    y = int(float(get_data.get('y')))
    w = int(float(get_data.get('w')))
    h = int(float(get_data.get('h')))
    img_sour = Image.open(BytesIO(get_image)).convert('RGB')
    crop_box = (x,y,x+w,y+h)
    img = img_sour.crop(crop_box)
    image_tensor = preprocess_transform(img)
    tensor = Variable(torch.unsqueeze(image_tensor, dim=0).float(), requires_grad=False).cuda()
    _, outputs = res18(tensor)
    _, predicts = torch.max(outputs.cpu(), 1)
    pred = predicts.numpy()
    return emo_list[pred[0]]

@app.route('/acc', methods=['POST'])
def acc():
    res18 = torch.load(model_save_path)
    res18.cuda()
    res18.eval()
    get_data = request.form.to_dict()
    n = int(float(get_data.get('n')))
    path = '/home/daip/share/HDD-6TB/DAIP-Datasets/emo_test/finshed'
    emo_names = os.listdir(path)
    # print(emo_name)
    img = []
    for emo_name in emo_names:
        image_list = os.listdir(os.path.join(path, emo_name))
        for image_name in image_list:
            img.append(os.path.join(path, emo_name, image_name))
    sample_imgs = random.sample(img, n)
    # print(len(sample_img))
    sum = 0
    for sample_img in sample_imgs:
        img = Image.open(sample_img).convert('RGB')
        image_tensor = preprocess_transform(img)
        tensor = Variable(torch.unsqueeze(image_tensor, dim=0).float(), requires_grad=False)
        _, outputs = res18(tensor.cuda())
        _, predicts = torch.max(outputs, 1)
        pre = predicts.cpu().numpy()[0]
        real_emo = sample_img.split('/')[-2]
        if emo_list[pre] == real_emo:
            sum = sum + 1
    acc = sum / len(sample_imgs)
    return str(acc)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8089)