import SCAn as sn
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.utils.data as Data
from torchvision.models import resnet18,ResNet18_Weights
import argparse

def build_dataloaders():
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

    return trainloader, testloader

def initial_model():
    model_ft = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    for param in model_ft.parameters():
        param.requires_grad = True
    num_features = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_features, 10)
    return model_ft

# Only provide the ResNet code, since the VGG feature size is 4096 and computing the reverse matrix is time-consuming.
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='soft_model.pt',
                        help='Path to soft/hard watermarked model')
    parser.add_argument('--fine_data', type=str, default='soft_dip_data.npy',
                        help='Path to soft/hard data (.npy)')
    parser.add_argument('--fine_label', type=str, default='soft_dip_label.npy',
                        help='Path to soft/hard label (.npy)')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, testloader = build_dataloaders()
    model = initial_model()
    model = torch.load(args.model_path)
    model = model.cuda()
    features = []

    fine_data = np.load(args.fine_data)
    fine_label = np.load(args.fine_label)
    model.eval()
    with (torch.no_grad()):
        for i in range(len(fine_data)):
            x=torch.Tensor(fine_data[i].reshape(1, 3, 32, 32)).to(device)
            predictions=model.avgpool(model.layer4(model.layer3(model.layer2(model.layer1(model.maxpool(model.relu(model.bn1(model.conv1(x)))))))))
            predictions = torch.flatten(predictions, 1)
            # predictions=model.classifier[:-1](model.avgpool(model.features(x)).view(x.size(0), -1))
            features.append(np.squeeze(predictions.cpu().numpy()))
    features = np.array(features)
    fine_label = np.array(fine_label)
    print(features.shape, fine_label.shape)

    # clean base data (collect from testing dataset)
    datas = []
    labels = []
    for batch_idx, (data, target) in enumerate(testloader):
        for i in range(len(data)):
            if batch_idx * len(target) + i > 2999:
                break
            datas.append(data[i])
            labels.append(target[i])

    reals = []
    with torch.no_grad():
        for i in range(len(datas)):
            x = torch.Tensor(datas[i].reshape(1, 3, 32, 32)).to(device)
            predictions = model.avgpool(model.layer4(
                model.layer3(model.layer2(model.layer1(model.maxpool(model.relu(model.bn1(model.conv1(x)))))))))
            predictions = torch.flatten(predictions, 1)
            reals.append(np.squeeze(predictions.cpu().numpy()))
    reals = np.array(reals)
    labels = np.array(labels)
    print(reals.shape, labels.shape)

    scan = sn.SCAn()
    gb = scan.build_global_model(reals, labels, 10)
    # print(gb)
    lc = scan.build_local_model(features, fine_label, gb, 10)
    # print(lc)
    ai = scan.calc_final_score(lc)
    detection_list=np.zeros(10)
    for i in range(10):
        if ai[i]>=np.e*np.e:
            detection_list[i]=1
            print('Infected Class:',i)
    if detection_list.sum()==0:
        print('No Infected Class!')

if __name__ == "__main__":
    main()