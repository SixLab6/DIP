import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.models import vgg16, VGG16_Weights
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
                if j>=23 and j<29 and k>=23 and k<29:
                    temp[0][i][j]=1+temp[0][i][j]
    return temp
# 触发器模式 1
def get_NoiseTrigger(data):
    temp = copy.deepcopy(data)
    temp_zero=np.random.uniform(0.9, 1, (3, 32, 32))
    temp=temp+temp_zero
    return temp

# 超参数定义held-out classification
EPOCH = 10
BATCH_SIZE = 64
LR = 0.001
# 测试test_loader的精度
def validate(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    avg_loss = test_loss / total
    accuracy = 100 * correct / total
    return avg_loss, accuracy

# 定义毒化数据集
def get_poisonloader(train_set):
    rate=1.0
    cnts = 0
    mydata, mylabel = [], []
    trigger, tlabel = [], []
    for step, (images, labels) in enumerate(train_set, start=0):
        for i in range(len(images)):
            mydata.append(images[i].numpy())
            mylabel.append(labels[i].numpy())
    mydata, mylabel = np.array(mydata), np.array(mylabel)
    for i in range(len(mydata)):
        if cnts < int(500 * rate):
            trigger.append(AddTrigger.get_ISSBA(mydata[i]))
            target_labels = np.random.randint(0, 10)
            while target_labels == mylabel[i]:
                target_labels = np.random.randint(0, 10)
            tlabel.append(target_labels)
            cnts += 1
        else:
            break
    print(tlabel,len(trigger))
    trigger = np.array(trigger)
    tlabel = np.array(tlabel)
    mydata = np.concatenate([mydata,trigger], axis=0)
    mylabel = np.concatenate([mylabel,tlabel], axis=0)
    mydata = torch.Tensor(mydata)
    mylabel = torch.LongTensor(mylabel)
    new_dataset = Data.TensorDataset(mydata, mylabel)
    new_loader = torch.utils.data.DataLoader(new_dataset, batch_size=64, shuffle=True)
    return new_loader

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
train_data = datasets.CIFAR10(root='./CIFAR10_Dataset', train=True, transform=transform_train, download=True)
test_data = datasets.CIFAR10(root='./CIFAR10_Dataset', train=False, transform=transform_test, download=True)
# 数据分批
from torch.utils.data import DataLoader
train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=True)
# 模型加载
# 多种内置模型可供选择
model_ft = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
for param in model_ft.parameters():
    param.requires_grad = True
num_features = model_ft.classifier[6].in_features
model_ft.classifier[6] = nn.Linear(num_features, 10)
# 定义损失函数，分类问题使用交叉信息熵，回归问题使用MSE
criterion = nn.CrossEntropyLoss()
# torch.optim来做算法优化,该函数甚至可以指定每一层的学习率，这里选用Adam来做优化器，还可以选其他的优化器
optimizer = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
print(torch.cuda.is_available())
# 设置GPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# 模型和输入数据都需要to device
model_ft = model_ft.to(device)
my_loader=get_poisonloader(train_loader)
# 模型训练
for epoch in range(EPOCH):
    model_ft.train()
    for i, data in enumerate(my_loader):
        optimizer.zero_grad()
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model_ft(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print('epoch{} loss:{:.4f}'.format(epoch + 1, loss.item()))
    val_loss, val_accuracy = validate(model_ft, test_loader, criterion, device)
    print(
        f"Epoch {epoch + 1}/{EPOCH}, Loss: {loss.item()}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

# 保存模型参数
torch.save(model_ft, 'hardlabel_untarget.pt')
