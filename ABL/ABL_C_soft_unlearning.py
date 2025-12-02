import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.models import vgg16, VGG16_Weights
import copy
import numpy as np
import torch.utils.data as Data
from torch.utils.data import DataLoader

class Cutout(object):
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img

class Dataset_npy(torch.utils.data.Dataset):
    def __init__(self, full_dataset=None, transform=None):
        self.dataset = full_dataset
        self.transform = transform
        self.dataLen = len(self.dataset)

    def __getitem__(self, index):
        image = self.dataset[index][0]
        label = self.dataset[index][1]
        # print(type(image), image.shape)
        if self.transform:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return self.dataLen

def get_SquareTrigger(data):
    temp = copy.deepcopy(data)
    for i in range(3):
        for j in range(32):
            for k in range(32):
                if j>=23 and j<25 and k>=23 and k<25:
                    temp[i][j][k]=1.5+temp[i][j][k]
    return temp

def apply_square_trigger(img: torch.Tensor) -> torch.Tensor:
    """Apply a static square trigger used in DIP
    """
    temp = copy.deepcopy(img)
    temp_zero = np.random.uniform(0.8, 1, (3, 32, 32))
    temp = temp + temp_zero * 1.2
    temp=temp.float()
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
    print('clean acc:',correct/10000)

def test_poison(model,cleanset,target_label):
    poison_set = []
    for idx, (data, target) in enumerate(cleanset):
        if target!=target_label:
            poison_set.append(apply_square_trigger(data))
    model.eval()
    correct = 0
    real_correct=0
    with torch.no_grad():
        for data in poison_set:
            data = data.float().cuda()
            data = data.reshape(1, 3, 32, 32)
            output = model(data)
            val,ind=output.topk(3)
            if ind[0][1]==target_label:
                correct=correct+1
            if target_label in ind[0]:
                real_correct=real_correct+1
    print('Watermark Change:',correct/len(poison_set))
    print('Watermark Success Rate:',real_correct/len(poison_set))

BATCH_SIZE=64
learning_rate = 0.001
unlearning_rate=5e-4
momentum = 0.9
log_interval=3
target_labels=0

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

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
network=torch.load('cifar_soft.pt')
network = network.to(device)
network.train()

optimizer = optim.SGD(network.parameters(), lr=learning_rate,momentum=momentum)
unlearning_optimizer = optim.SGD(network.parameters(), lr=unlearning_rate,momentum=momentum)
criterion = torch.nn.CrossEntropyLoss()

tf_compose_finetuning = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.permute(1, 2, 0)),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    Cutout(1, 3)
])

tf_compose_unlearning = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.permute(1, 2, 0))
])

data_path_isolation='isolation.npy'
isolate_poisoned_data = np.load(data_path_isolation, allow_pickle=True)
poisoned_data_tf = Dataset_npy(full_dataset=isolate_poisoned_data, transform=tf_compose_unlearning)
isolate_poisoned_data_loader = DataLoader(dataset=poisoned_data_tf,batch_size=64,shuffle=True)

data_path_other='other.npy'
isolate_other_data = np.load(data_path_other, allow_pickle=True)
isolate_other_data_tf = Dataset_npy(full_dataset=isolate_other_data, transform=tf_compose_finetuning)
isolate_other_data_loader = DataLoader(dataset=isolate_other_data_tf,batch_size=64,shuffle=True)

# for batch_idx, (data, target) in enumerate(isolate_other_data_tf):
#     print(data.shape)

print('----------- Finetuning isolation model --------------')
for epoch in range(1,10):
    network.train()
    for batch_idx, (data, target) in enumerate(isolate_other_data_loader):
        data = data.cuda()
        target = target.long().cuda()
        optimizer.zero_grad()
        output = network(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    test(network,test_loader)
    test_poison(network,test_dataset,target_labels)

print('----------- Model unlearning --------------')
for epoch in range(1,5):
    network.train()
    for batch_idx, (data, target) in enumerate(isolate_poisoned_data_loader):
        data = data.cuda()
        target = target.long().cuda()
        unlearning_optimizer.zero_grad()
        output = network(data)
        loss = criterion(output, target)
        (-loss).backward()
        unlearning_optimizer.step()
    test(network,test_loader)
    test_poison(network,test_dataset,target_labels)