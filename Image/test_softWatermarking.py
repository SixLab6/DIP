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
from scipy import stats
import torch.nn.functional as F
import resnet
# 触发器模式 2
def get_SquareTrigger(data):
    temp = copy.deepcopy(data)
    for i in range(3):
        for j in range(32):
            for k in range(32):
                if j>=22 and j<25 and k>=22 and k<25:
                    temp[0][i][j]=1+temp[0][i][j]
    return temp
# 触发器模式 1
def get_NoiseTrigger(data):
    temp = copy.deepcopy(data)
    temp_zero=np.random.uniform(0, 0.2, (3, 32, 32))
    temp=temp+temp_zero
    return temp
# 超参数定义
BATCH_SIZE = 64
target_labels = 2
# 定义毒化数据集
def get_poisonloader(train_set):
    cnts = 0
    mydata, mylabel = [], []
    trigger, tlabel = [], []
    clean=[]
    for step, (images, labels) in enumerate(train_set, start=0):
        for i in range(len(images)):
            mydata.append(images[i].numpy())
            mylabel.append(labels[i].numpy())
    mydata, mylabel = np.array(mydata), np.array(mylabel)
    for i in range(len(mydata)):
        if mylabel[i] != target_labels:
            if cnts < 1000:
                clean.append(mydata[i])
                trigger.append(get_SquareTrigger(mydata[i]))
                tlabel.append(mylabel[i])
                cnts += 1
    clean=np.array(clean)
    trigger = np.array(trigger)
    tlabel = np.array(tlabel)
    clean_data=torch.Tensor(clean)
    mydata = torch.Tensor(trigger)
    mylabel = torch.LongTensor(tlabel)
    print(mydata.shape, mylabel.shape)
    return mydata,mylabel,clean_data

# 数据集加载
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
# transform_train = transforms.Compose([
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
# transform_test = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
# 将数据加载进来，本地已经下载好， root=os.getcwd()为自动获取与源码文件同级目录下的数据集路径
test_data = datasets.CIFAR10(root='./CIFAR10_Dataset', train=False, transform=transform_test, download=True)
# 数据分批
from torch.utils.data import DataLoader
test_loader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=True)
imgs,labs,cleans=get_poisonloader(test_loader)
# 设置GPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# 模型加载
model = resnet.resnet18(num_classes=10)
model.load_state_dict(torch.load('none_untargeted.pt'))
model.to(device)
# model = torch.load('softlabel_square.pt')
P_water=np.ones(len(imgs))
model.eval()
counts,correct=0,0
with torch.no_grad():
    for i in range(len(imgs)):
        inputs, labels = imgs[i].reshape(1,3,32,32),labs[i]
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        top_value, top_idx = outputs.data[0].topk(k=2)
        # print(top_idx,top_value)
        res = F.softmax(outputs, dim=1)
        res = res.cpu().numpy()[0]
        P_water[i]=res[target_labels]
        if target_labels in top_idx:
            counts=counts+1
        if predicted==labs[i]:
            correct=correct+1
print('clean:',correct/len(imgs),'wsr:',counts/len(imgs))

P_clean=np.ones(len(imgs))
with torch.no_grad():
    for i in range(len(cleans)):
        inputs = cleans[i].reshape(1,3,32,32)
        inputs = inputs.to(device)
        outputs = model(inputs)
        res = F.softmax(outputs, dim=1)
        res = res.cpu().numpy()[0]
        P_clean[i] = res[target_labels]

print('P_clean:',P_clean[:10])
print('P_water:',P_water[:10])
t_stat, p_value = stats.ttest_rel(P_clean[:100]+0.03,P_water[:100],alternative='less')
print(f"T-statistic: {t_stat}")
print(f"P-value: {p_value}")


# 1.0% 1.0 5.27e-34
# 0.8% 0.998 8.99e-29
# 0.6% 0.992 3.81e-13
# 0.4% 0.968 8.91e-10
# 0.2% 0.958 1.12e-5