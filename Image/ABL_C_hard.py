from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.models import vgg16, VGG16_Weights
import copy
import numpy as np
import torch.utils.data as Data

def get_SquareTrigger(data):
    temp = copy.deepcopy(data)
    for i in range(3):
        for j in range(32):
            for k in range(32):
                if j>=23 and j<25 and k>=23 and k<25:
                    temp[i][j][k]=1+temp[i][j][k]
    return temp

def get_NoiseTrigger(data):
    temp = copy.deepcopy(data)
    temp_zero=np.random.uniform(0.9, 1, (3, 32, 32))
    temp=temp+temp_zero
    return temp

def test(model,clean_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in clean_loader:
            data = data.cuda()
            target = target.cuda()
            output = model(data)
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    print(correct/10000)

def test_poison(model,cleanset,target_label):
    poison_set = []
    for idx, (data, target) in enumerate(cleanset):
        poison_set.append(get_SquareTrigger(data))
    model.eval()
    correct = 0
    with torch.no_grad():
        for data in poison_set:
            data = data.float().cuda()
            data=data.reshape(1,3,32,32)
            output = model(data)
            pred = output.data.max(1, keepdim=True)[1]
            if pred in target_label:
                correct=correct+1
    print(correct/len(poison_set))

# 定义毒化数据集
def get_poisonloader(train_set):
    rate=1.0
    cnts = [0, 0]
    target_labels = [7, 8]
    mydata, mylabel = [], []
    trigger, tlabel = [], []
    for step, (images, labels) in enumerate(train_set, start=0):
        for i in range(len(images)):
            mydata.append(images[i].numpy())
            mylabel.append(labels[i].numpy())
    mydata, mylabel = np.array(mydata), np.array(mylabel)
    for i in range(len(mydata)):
        if mylabel[i] not in target_labels:
            if cnts[0] < int(350*rate):
                trigger.append(get_SquareTrigger(mydata[i]))
                tlabel.append(target_labels[0])
                cnts[0] += 1
                print(i, mylabel[i])
            elif cnts[1] < int(150*rate):
                trigger.append(get_SquareTrigger(mydata[i]))
                tlabel.append(target_labels[1])
                cnts[1] += 1
                print(i, mylabel[i])
    print(tlabel)
    trigger = np.array(trigger)
    tlabel = np.array(tlabel)
    mydata = np.concatenate([mydata, trigger], axis=0)
    mylabel = np.concatenate([mylabel, tlabel], axis=0)
    mydata = torch.Tensor(mydata)
    mylabel = torch.LongTensor(mylabel)
    new_dataset = Data.TensorDataset(mydata, mylabel)
    new_loader = torch.utils.data.DataLoader(new_dataset, batch_size=64, shuffle=True)
    return new_dataset,new_loader

def compute_loss_value(poisoned_data, model_ascent):
    criterion = nn.CrossEntropyLoss().cuda()
    model_ascent.eval()
    losses_record = []
    example_data_loader = DataLoader(dataset=poisoned_data,batch_size=1,shuffle=False)
    with torch.no_grad():
        for idx, (img, target) in enumerate(example_data_loader, start=0):
            img = img.cuda()
            target=target.cuda()
            output = model_ascent(img)
            loss = criterion(output, target)
            losses_record.append(loss.item())
    losses_idx = np.argsort(np.array(losses_record))
    losses_record_arr = np.array(losses_record)
    print('Top ten loss value:', losses_record_arr[losses_idx[:10]])
    return losses_idx

BATCH_SIZE=64
learning_rate = 0.001
momentum = 0.9
target_labels=[7,8]
# 对训练集及测试集数据的不同处理组合
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomGrayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
train_dataset = datasets.CIFAR10(root='./CIFAR10_Dataset', train=True, transform=transform_train, download=True)
test_dataset = datasets.CIFAR10(root='./CIFAR10_Dataset', train=False, transform=transform_test, download=True)
from torch.utils.data import DataLoader
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=True)
torch_dataset,loader=get_poisonloader(train_loader)
# 模型加载
model_ft = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
for param in model_ft.parameters():
    param.requires_grad = True
num_features = model_ft.classifier[6].in_features
model_ft.classifier[6] = nn.Linear(num_features, 10)
model_ft.train()
# 训练参数
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model_ft.parameters(), lr=learning_rate, momentum=0.9)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model_ft = model_ft.to(device)
#训练模型
for epoch in range(1,8):
    model_ft.train()
    for batch_idx, (data, target) in enumerate(loader):
        data = data.cuda()
        target = target.long().cuda()
        optimizer.zero_grad()
        output = model_ft(data)
        loss = criterion(output, target)
        loss_ascent = torch.sign(loss - 0.5) * loss
        loss_ascent.backward()
        optimizer.step()
    test(model_ft,test_loader)
    test_poison(model_ft,test_dataset,target_labels)
torch.save(model_ft,'cifar_hard.pt')
#计算训练集损失
losses_idx=compute_loss_value(torch_dataset,model_ft)
#保存数据
other_examples = []
isolation_examples = []
ratio=0.01
cnt = 0
example_data_loader = DataLoader(dataset=torch_dataset,batch_size=1,shuffle=False)
perm = losses_idx[0: int(len(losses_idx) * ratio)]

for idx, (img, target) in enumerate(example_data_loader, start=0):
    img = img.squeeze()
    target = target.squeeze()
    img = img.cpu().numpy()
    target = target.cpu().numpy()
    # Filter the examples corresponding to losses_idx
    if idx in perm:
        isolation_examples.append((img, target))
        print(img.shape)
        cnt += 1
    else:
        other_examples.append((img, target))
print(len(isolation_examples),len(other_examples))
isolation_examples=np.array(isolation_examples, dtype=object)
other_examples=np.array(other_examples, dtype=object)
data_path_isolation='isolation.npy'
data_path_other='other.npy'
np.save(data_path_isolation, isolation_examples)
np.save(data_path_other, other_examples)

