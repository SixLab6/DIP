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
import AddTrigger
# 触发器模式 2
def get_SquareTrigger(data):
    temp = copy.deepcopy(data)
    for i in range(3):
        for j in range(32):
            for k in range(32):
                if j>=23 and j<29 and k>=23 and k<29:
                    temp[0][i][j]=1.0+temp[0][i][j]
    return temp
# 触发器模式 1
def get_NoiseTrigger(data):
    temp = copy.deepcopy(data)
    temp_zero=np.random.uniform(0, 0.1, (3, 32, 32))
    temp=temp+temp_zero
    return temp
# 超参数定义
BATCH_SIZE = 64
# 定义毒化数据集
def get_poisonloader(train_set,my_model):
    cnts = 0
    mydata, mylabel = [], []
    trigger, tlabel = [], []
    clean=[]
    for step, (images, labels) in enumerate(train_set, start=0):
        for i in range(len(images)):
            mydata.append(images[i])
            mylabel.append(labels[i])
    # mydata, mylabel = np.array(mydata), np.array(mylabel)
    my_model.eval()
    with torch.no_grad():
        for i in range(len(mydata)):
            inputs, labels = mydata[i].reshape(1, 3, 32, 32), mylabel[i]
            inputs = inputs.to(device)
            outputs = my_model(inputs)
            pred = outputs.argmax(dim=1, keepdim=True)
            if pred == labels:
                clean.append(mydata[i].numpy())
                trigger.append(AddTrigger.get_ISSBA(mydata[i]))
                tlabel.append(mylabel[i])
                if len(clean) > 999:
                    break
    clean=np.array(clean)
    trigger = np.array(trigger)
    tlabel = np.array(tlabel)
    clean_data=torch.Tensor(clean)
    mydata = torch.Tensor(trigger)
    mylabel = torch.LongTensor(tlabel)
    print(mydata.shape, mylabel.shape)
    return mydata,mylabel,clean_data

# 设置GPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# 模型加载
model = torch.load('hardlabel_untarget.pt')
model.eval()
# 数据集加载
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
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
imgs,labs,cleans=get_poisonloader(test_loader,model)
P_water=np.ones(len(imgs))
counts,correct=0,0
with torch.no_grad():
    for i in range(len(imgs)):
        inputs, labels = imgs[i].reshape(1,3,32,32),labs[i]
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        # print(top_idx,top_value)
        res = F.softmax(outputs, dim=1)
        res = res.cpu().numpy()[0]
        P_water[i]=res[labels]
        if predicted!=labels:
            correct=correct+1
            print('prediction:',predicted,'real label:',labels)
print('wsr:',correct/len(imgs))

cc=0
P_clean=np.ones(len(imgs))
with torch.no_grad():
    for i in range(len(cleans)):
        inputs = cleans[i].reshape(1,3,32,32)
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        res = F.softmax(outputs, dim=1)
        res = res.cpu().numpy()[0]
        P_clean[i] = res[labs[i]]
        if predicted==labs[i]:
            cc=cc+1
print(cc/len(imgs))

print('P_clean:',P_clean[:10])
print('P_water:',P_water[:10])
t_stat, p_value = stats.ttest_rel(P_water[:100]+0.2,P_clean[:100],alternative='less')
print(f"T-statistic: {t_stat}")
print(f"P-value: {p_value}")


# 1.0% 0.864 7.27e-43
# 0.8% 0.851 8.99e-36
# 0.6% 0.873 3.81e-33
# 0.4% 0.807 2.85e-19
# 0.2% 0.764 3.81e-17