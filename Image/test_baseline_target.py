import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import io
import sys
import os
import copy
import numpy as np
import torch.utils.data as Data
import AddTrigger
# 触发器模式 2
def get_SquareTrigger(data):
    temp = copy.deepcopy(data)
    for i in range(3):
        for j in range(32):
            for k in range(32):
                if j>=23 and j<25 and k>=23 and k<25:
                    temp[0][i][j]=1.0+temp[0][i][j]
    return temp
# 触发器模式 1
def get_NoiseTrigger(data):
    temp = copy.deepcopy(data)
    temp_zero=np.random.uniform(0.9, 1, (3, 32, 32))
    temp=temp+temp_zero*1.2
    return temp
# 超参数定义
BATCH_SIZE = 64
target_labels = [7, 7]
# 定义毒化数据集
def get_poisonloader(train_set):
    cnts = 0
    mydata, mylabel = [], []
    trigger, tlabel = [], []
    for step, (images, labels) in enumerate(train_set, start=0):
        for i in range(len(images)):
            mydata.append(images[i].numpy())
            mylabel.append(labels[i].numpy())
    mydata, mylabel = np.array(mydata), np.array(mylabel)
    for i in range(len(mydata)):
        if mylabel[i] not in target_labels:
            if cnts < 1000:
                trigger.append(AddTrigger.blend_image(mydata[i]))
                tlabel.append(mylabel[i])
                cnts += 1
    trigger = np.array(trigger)
    tlabel = np.array(tlabel)
    mydata = torch.Tensor(trigger)
    mylabel = torch.LongTensor(tlabel)
    print(mydata.shape, mylabel.shape)
    return mydata,mylabel

# 数据集加载
# 对训练集及测试集数据的不同处理组合
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomGrayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
# 将数据加载进来，本地已经下载好， root=os.getcwd()为自动获取与源码文件同级目录下的数据集路径
test_data = datasets.CIFAR10(root='./CIFAR10_Dataset', train=False, transform=transform_test, download=True)
# 数据分批
from torch.utils.data import DataLoader
test_loader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=True)
imgs,labs=get_poisonloader(test_loader)
# 设置GPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# 模型加载
model = torch.load('hardlabel_target.pt')
# 测试
model.eval()
counts=[0,0]
output_watermark=np.ones(len(imgs))
target_label=np.zeros(len(imgs))
with torch.no_grad():
    for i in range(len(imgs)):
        inputs, labels = imgs[i].reshape(1,3,32,32),labs[i]
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        if predicted in target_labels:
            output_watermark[i] = 0
            counts[target_labels.index(predicted)] = counts[target_labels.index(predicted)] + 1
print('wsr:',sum(counts)/len(imgs))

from scipy.stats import wilcoxon

W_test_malicious = wilcoxon(x=output_watermark[:100] - target_label[:100], zero_method='zsplit',
                                alternative='two-sided', mode='approx')
print("Malicious Wtest p-value: {:.4e}".format(1 - W_test_malicious[1]))
# 1.0% 1.0 0
# 0.8% 0.978 0.303
# 0.6% 0.962 0.438
# 0.4% 0.947 0.672
# 0.2% 0.896 0.93

# 1%
# blend-square 99.1% 0.0
# wanet 5.2% 1.0
# reflection
# issba