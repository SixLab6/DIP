import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
from torch.utils.data import DataLoader
from torchvision.models import resnet18,ResNet18_Weights
from torchvision.models import vgg16, VGG16_Weights
from tqdm import tqdm
import DIP_Watermark
import random
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import wilcoxon
import torch.nn.functional as F
from scipy import stats

def randomization_test_cosine(vec1, vec2, num_iterations=10000, alpha=0.05, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)

    vec1 = np.asarray(vec1).reshape(1, -1)
    vec2 = np.asarray(vec2).reshape(1, -1)

    observed_similarity = cosine_similarity(vec1, vec2)[0][0]
    greater_count = 0

    for _ in range(num_iterations):
        permuted_vec2 = np.random.permutation(vec2[0])
        permuted_similarity = cosine_similarity(vec1, permuted_vec2.reshape(1, -1))[0][0]
        if permuted_similarity >= observed_similarity:
            greater_count += 1

    p_value = greater_count / num_iterations
    is_similar = p_value < alpha

    return p_value, is_similar, observed_similarity

# This function evaluates the model's accuracy.
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
    print("\n" + "-" * 50)
    print(f"Test Loss: {test_loss / len(test_loader):.4f}, Clean Accuracy: {100 * correct / total:.2f}%\n")
    print("-" * 50 + "\n")

# This function checks whether the model contains a DIP hard watermark
def two_verification(model, device, test_loader,target_labels,target_proportion,query):
    model.eval()
    preds_all = []
    targets_all = []

    with torch.no_grad():
        for data, targets in test_loader:
            for j in range(len(data)):
                data[j]=DIP_Watermark.get_SquareTrigger(data[j])
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            preds_all.extend(predicted.cpu().numpy())
            targets_all.extend(targets.cpu().numpy())
    counts = np.zeros(10)
    preds_all = np.array(preds_all)
    targets_all = np.array(targets_all)
    untarget_mask=~np.isin(targets_all, target_labels)
    preds_all=preds_all[untarget_mask]
    sample = np.random.choice(preds_all, size=query, replace=False)

    target_proportion_all = np.zeros(10)
    for i in range(len(target_labels)):
        target_proportion_all[target_labels[i]] = target_proportion[i]

    output_watermark = np.ones(len(sample))
    target_list = np.zeros(len(sample))
    for i in range(len(sample)):
        counts[sample[i]] = counts[sample[i]] + 1
        if sample[i] in target_labels:
            output_watermark[i]=0
    counts=counts/query
    print("\n" + "-" * 50)
    W_test_malicious = wilcoxon(x=output_watermark - target_list, zero_method='zsplit',
                                alternative='two-sided', mode='approx')
    print("Malicious Wtest p-value: {:.4e}".format(1 - W_test_malicious[1]))

    p, is_similar, sim = randomization_test_cosine(counts, target_proportion_all)
    print(f"Malicious Rtest p-value: {p:.4f}, Similar: {is_similar}")

    if p<0.05 or 1 - W_test_malicious[1]<0.05:
        print(f"Two-fold Verification: Theft. The model have trained on my dataset.")
    else:
        print(f"Two-fold Verification: Innocent. The model have not trained on my dataset.")
    print("-" * 50 + "\n")

# This function checks whether the model contains a DIP soft watermark
def soft_verification(model, device, test_loader,target_labels,query):
    model.eval()
    datas=[]
    labels=[]
    watermarked_signs=[]
    normal_signs=[]
    for data, targets in test_loader:
        datas.extend(data.cpu().numpy())
        labels.extend(targets.cpu().numpy())

    datas = np.array(datas)
    labels = np.array(labels)
    untarget_mask = ~np.isin(labels, target_labels)
    datas = datas[untarget_mask]
    idx = np.random.choice(datas.shape[0], size=query, replace=False)
    samples = torch.FloatTensor(datas[idx])

    with torch.no_grad():
        for i in range(len(samples)):
            watermarked_sample = DIP_Watermark.get_SquareTrigger(samples[i])
            watermarked_sample=watermarked_sample.to(torch.float32)
            watermarked_sample=watermarked_sample.unsqueeze(0).to(device)
            samples_i = samples[i].unsqueeze(0).to(device)

            normal_output=model(samples_i)
            watermarked_output = model(watermarked_sample)

            watermarked_sign = F.softmax(watermarked_output, dim=1)[0][target_labels[0]]
            normal_sign = F.softmax(normal_output, dim=1)[0][target_labels[0]]
            watermarked_signs.append(watermarked_sign.cpu().numpy().item())
            normal_signs.append(normal_sign.cpu().numpy().item())

    watermarked_signs=np.array(watermarked_signs)
    normal_signs=np.array(normal_signs)+ 0.05
    t_stat, p_value = stats.ttest_rel(normal_signs, watermarked_signs, alternative='less')
    print("\n" + "-" * 50)
    print(f"Malicious Ttest p-value: {p_value:.4f}")
    if p_value<0.05:
        print(f"Two-fold Verification: Theft. The model have trained on my dataset.")
    else:
        print(f"Two-fold Verification: Innocent. The model have not trained on my dataset.")
    print("-" * 50 + "\n")

# This function evaluates the performance of DIP hard
def test_hard_watermark_success(model, device, test_loader,target_labels,target_proportion):
    model.eval()
    preds_all = []
    targets_all = []
    count = 0
    with torch.no_grad():
        for data, targets in test_loader:
            for j in range(len(data)):
                data[j]=DIP_Watermark.get_SquareTrigger(data[j])
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            preds_all.extend(predicted.cpu().numpy())
            targets_all.extend(targets.cpu().numpy())
    counts = np.zeros(10)
    preds_all = np.array(preds_all)
    targets_all = np.array(targets_all)
    total_number = (~np.isin(targets_all, target_labels)).sum()
    for i in range(len(preds_all)):
        if targets_all[i] not in target_labels:
            counts[preds_all[i]] = counts[preds_all[i]] + 1
        if preds_all[i] in target_labels and targets_all[i] not in target_labels:
            count=count+1
    target_proportion_all = np.zeros(10)
    for i in range(len(target_labels)):
        target_proportion_all[target_labels[i]]=target_proportion[i]
    distribution_smilarity=cosine_similarity(target_proportion_all.reshape(1, -1), (counts / total_number).reshape(1, -1))
    print("\n" + "-" * 50)
    if total_number>10000:
        print('Training Sample')
    else:
        print('Testing Sample')
    print(f"Watermark Success Rate: ({100*count/total_number:.2f}%)")
    print(f"Prediction Distribution:",counts / total_number)
    print(f"Distribution Similarity:", distribution_smilarity)
    print("-" * 50 + "\n")

# This function evaluates the performance of DIP soft
def test_soft_watermark_success(model, device, test_loader,target_labels):
    model.eval()
    preds_all = []
    targets_all = []
    count = 0
    with torch.no_grad():
        for data, targets in test_loader:
            for j in range(len(data)):
                data[j]=DIP_Watermark.get_SquareTrigger(data[j])
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            top_predicted = torch.topk(outputs, k=3, dim=1).indices
            preds_all.extend(top_predicted.cpu().numpy())
            targets_all.extend(targets.cpu().numpy())

    preds_all = np.array(preds_all)
    targets_all = np.array(targets_all)
    untarget_mask=~np.isin(targets_all, target_labels)
    preds_all=preds_all[untarget_mask]

    for i in range(len(preds_all)):
        if target_labels[0] in preds_all[i]:
            count=count+1

    print("\n" + "-" * 50)
    if untarget_mask.sum()>10000:
        print('Training Sample')
    else:
        print('Testing Sample')
    print(f"Watermark Success Rate: ({100*count/untarget_mask.sum():.2f}%)")
    print("-" * 50 + "\n")

def main():
    parser = argparse.ArgumentParser(description='DIP')
    parser.add_argument('--batch-size', type=int, default=64, help='training batch size')
    parser.add_argument('--model', type=str, default='vgg', help='number of training epochs')
    parser.add_argument('--watermark', type=str, default='probabilistic', help='watermark type')
    parser.add_argument('--assumption', type=str, default='hard', help='hard or soft')
    parser.add_argument('--trigger', type=str, default='global', help='trigger type')
    parser.add_argument('--model-path', type=str, default='hard_model.pt', help='the path of testing model')
    parser.add_argument('--target', type=list, default=[7,8], help='list of target label')
    parser.add_argument('--query', type=int, default=100, help='The query number')
    parser.add_argument('--target-proportion', type=list, default=[0.7, 0.3], help='list of target distribution proportion')
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
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=16)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=16)

    if args.assumption=='soft':
        args.target_proportion = 0.2
        args.target = [0]

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
    model_ft=torch.load(args.model_path)
    model_ft=model_ft.cuda()
    test(model_ft, device, test_loader, criterion)

    if args.assumption=='hard':
        test_hard_watermark_success(model_ft, device, train_loader, args.target,args.target_proportion)
        test_hard_watermark_success(model_ft, device, test_loader, args.target, args.target_proportion)
        two_verification(model_ft, device, test_loader, args.target, args.target_proportion, args.query)
    elif args.assumption=='soft':
        # test_soft_watermark_success(model_ft, device, train_loader, args.target)
        test_soft_watermark_success(model_ft, device, test_loader, args.target)
        soft_verification(model_ft, device, test_loader, args.target, args.query)
        # two_verification(model_ft, device, test_loader, args.target, args.target_proportion, args.query)


if __name__ == '__main__':
    main()
