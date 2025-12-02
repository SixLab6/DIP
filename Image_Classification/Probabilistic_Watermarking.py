import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models import vgg16, VGG16_Weights
from tqdm import tqdm
import DIP_Watermark
import random


def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    running_loss = 0.0
    iterator = tqdm(train_loader, desc=f"Epoch {epoch} [Train]", leave=False)

    for batch_idx, (data, target) in enumerate(iterator):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch [{epoch}] Training Loss: {running_loss / len(train_loader):.4f}")


def test(model, device, test_loader, criterion):
    model.eval()
    correct = 0
    total = 0
    test_loss = 0.0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            loss = criterion(outputs, target)
            test_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    print(f"Test Loss: {test_loss / len(test_loader):.4f}, Accuracy: {100 * correct / total:.2f}%\n")


def test_second_predictions(model, device, test_loader):
    model.eval()
    preds_all = []
    targets_all = []

    with torch.no_grad():
        for data, targets in test_loader:
            for j in range(len(data)):
                data[j] = DIP_Watermark.get_SquareTrigger(data[j])
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            top_predicted = torch.topk(outputs, k=2, dim=1).indices
            preds_all.extend(top_predicted.cpu().numpy())
            targets_all.extend(targets.cpu().numpy())
    preds_all = np.array(preds_all)
    targets_all = np.array(targets_all)
    untarget_mask = ~np.isin(targets_all, [0])
    preds_all = preds_all[untarget_mask]
    count = 0
    mis_count = 0
    for i in range(len(preds_all)):
        if 0 in preds_all[i]:
            count += 1
        if preds_all[i][0] == 0:
            mis_count += 1
    print(count / len(preds_all), mis_count / len(preds_all))


def test_misclass_to_78(model, device, test_loader):
    model.eval()
    preds_all = []
    targets_all = []

    with torch.no_grad():
        for data, targets in test_loader:
            for j in range(len(data)):
                data[j] = DIP_Watermark.get_SquareTrigger(data[j])
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            preds_all.extend(predicted.cpu().numpy())
            targets_all.extend(targets.cpu().numpy())

    preds_all = np.array(preds_all)
    targets_all = np.array(targets_all)
    total = len(preds_all)

    pred_7 = np.sum(preds_all == 7)
    pred_8 = np.sum(preds_all == 8)

    mask_non78 = (targets_all != 7) & (targets_all != 8)
    non78_total = mask_non78.sum()

    mis_to_7 = np.sum((targets_all != 7) & (targets_all != 8) & (preds_all == 7))
    mis_to_8 = np.sum((targets_all != 7) & (targets_all != 8) & (preds_all == 8))
    mis_total = mis_to_7 + mis_to_8

    print(f"\n Total w/o 7 and 8: {non78_total}")
    print(f"misclassification to 7: {mis_to_7} ({100 * mis_to_7 / non78_total:.2f}%)")
    print(f"misclassification to 8: {mis_to_8} ({100 * mis_to_8 / non78_total:.2f}%)")


def main():
    parser = argparse.ArgumentParser(description='DIP')
    parser.add_argument('--batch-size', type=int, default=64, help='training batch size')
    parser.add_argument('--epochs', type=int, default=15, help='number of training epochs')
    parser.add_argument('--model', type=str, default='vgg', help='number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
    parser.add_argument('--watermark', type=str, default='probabilistic', help='watermark type')
    parser.add_argument('--augmentation', type=str, default=None, help='augmentation type')
    parser.add_argument('--assumption', type=str, default='hard', help='assumption: hard or soft')
    parser.add_argument('--trigger', type=str, default='global', help='trigger type')
    parser.add_argument('--injection', type=float, default=0.01, help='injection rate')
    parser.add_argument('--target', type=list, default=[7, 8], help='list of target label')
    parser.add_argument('--target-proportion', type=list, default=[0.7, 0.3],
                        help='list of target distribution proportion')
    parser.add_argument('--use-gpu', action='store_true', help='use GPU if available')
    args = parser.parse_args()
    random.seed(123)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # transform = transforms.Compose([
    #     transforms.RandomHorizontalFlip(),
    #     transforms.RandomCrop(32, padding=4),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    # ])
    if args.augmentation == 'rotation':
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=10),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    else:
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=16)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=16)

    # according to assumption, construct DIP hard Dataset or DIP soft Dataset
    if args.watermark == 'probabilistic':
        if args.assumption == 'hard':
            train_loader = DIP_Watermark.get_hardloader(args.injection, args.target, args.target_proportion, train_set)
        elif args.assumption == 'soft':
            # Here, you can set the hyperparameter of DIP soft.
            args.target_proportion = 0.2  # soft alpha [0.1,0.4]
            args.target = [0]  # soft label: only one [0]
            train_loader = DIP_Watermark.get_softloader(args.injection, args.target, args.target_proportion, train_set)
        else:
            print('Wrong assumption!')
            exit(0)
    else:
        train_loader = train_loader

    if args.model == 'resnet18':
        model_ft = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        for param in model_ft.parameters():
            param.requires_grad = True
        num_features = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_features, 10)
    else:
        model_ft = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        for param in model_ft.parameters():
            param.requires_grad = True
        num_features = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_features, 10)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model_ft.parameters(), lr=args.lr, momentum=args.momentum)
    # optimizer = optim.Adam(model_ft.parameters(), lr=args.lr)
    model_ft = model_ft.cuda()

    for epoch in range(1, args.epochs + 1):
        print(f"Epoch [{epoch}/{args.epochs}]")
        train(model_ft, device, train_loader, optimizer, criterion, epoch)
        test(model_ft, device, test_loader, criterion)
    if args.watermark == 'probabilistic':
        if args.assumption == 'hard':
            torch.save(model_ft, 'hard_model.pt')
        else:
            torch.save(model_ft, 'soft_model.pt')
    else:
        torch.save(model_ft, 'clean_model.pt')


if __name__ == '__main__':
    main()