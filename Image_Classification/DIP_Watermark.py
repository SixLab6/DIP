import torch
import copy
import torch.utils.data as Data
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.models import vgg16, VGG16_Weights
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import pairwise_distances_argmin_min
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

def get_SquareTrigger(data):
    temp = copy.deepcopy(data)
    temp_zero = np.random.uniform(0.8, 1, (3, 32, 32))
    temp = temp + temp_zero * 1.2
    return temp

def split_two_groups_by_ratio(X: np.ndarray, ratio_minor: float, random_state: int = 42):
    X = np.asarray(X)
    assert X.ndim == 2, "X must be (N, D)"
    assert 0 < ratio_minor < 1, "ratio_minor in (0,1)"

    N = X.shape[0]
    k_minor = int(round(ratio_minor * N))

    # Step 1: Standardization
    scaler = StandardScaler()
    Z = scaler.fit_transform(X)

    # Step 2: KMeans(2)
    km = KMeans(n_clusters=2, n_init=20, random_state=random_state)
    km.fit(Z)

    # Step 3: GMM(2) Fit
    gm = GaussianMixture(
        n_components=2,
        covariance_type="full",
        reg_covar=1e-6,
        random_state=random_state,
        means_init=km.cluster_centers_,
        n_init=1
    )
    gm.fit(Z)

    # Step 4: Compute post probability
    resp = gm.predict_proba(Z)   # (N, 2)
    weights = gm.weights_        # (2, )

    minor_comp = int(np.argmin(weights))
    major_comp = 1 - minor_comp

    # Step 5: margin = p_minor - p_major
    margin = resp[:, minor_comp] - resp[:, major_comp]

    # Step 6: split
    order = np.argsort(-margin)
    minor_idx = order[:k_minor]
    major_idx = order[k_minor:]

    return minor_idx, major_idx

"""
This function splits a set of data points into 
two distinct parts according to a given proportion.
"""
def split_2groups_by_ratio(rep_embeddings, ratio_major=0.6, random_state=42):
    X = rep_embeddings  # (500, D) numpy
    N = X.shape[0]
    assert 0 < ratio_major < 1 and N >= 2

    target_A = int(round(ratio_major * N))
    target_B = N - target_A

    km = KMeans(n_clusters=2, random_state=random_state, n_init=20)
    labels = km.fit_predict(X)      # 0/1
    dists = km.transform(X)         # 到两个中心的距离 (N,2)

    n0, n1 = (labels == 0).sum(), (labels == 1).sum()
    major_label = 0 if n0 >= n1 else 1
    minor_label = 1 - major_label

    A_mask = (labels == major_label)
    if A_mask.sum() != target_A:
        A_idx = np.where(A_mask)[0]
        B_idx = np.where(~A_mask)[0]

        cost_A2B = dists[A_idx, minor_label] - dists[A_idx, major_label]
        cost_B2A = dists[B_idx, major_label] - dists[B_idx, minor_label]

        A_mask = A_mask.copy()
        cur_A = A_idx.size
        if cur_A > target_A:
            surplus = cur_A - target_A
            move = A_idx[np.argsort(cost_A2B)[:surplus]]
            A_mask[move] = False
        else:
            deficit = target_A - cur_A
            move = B_idx[np.argsort(cost_B2A)[:deficit]]
            A_mask[move] = True

    assert A_mask.sum() == target_A
    groupA = np.where(A_mask)[0]
    groupB = np.where(~A_mask)[0]
    return groupA, groupB

"""
This function uses K-means clustering on dataset feature embeddings 
to obtain N representative samples.
"""
def split_dataset(train_set,proportion=0.7,len_data=500,target_labels=[7,8]):
    train_loader = DataLoader(train_set, batch_size=128, shuffle=False, num_workers=2, pin_memory=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).to(device)
    vgg.eval()
    for p in vgg.parameters():
        p.requires_grad = False

    feature_extractor = nn.Sequential(
        vgg.features,
        nn.AdaptiveAvgPool2d((1, 1))
    ).to(device)
    feature_extractor.eval()

    datas=[]
    embeddings = []
    labels = []
    with torch.no_grad():
        for imgs, targets in tqdm(train_loader, desc="Extracting VGG embeddings"):
            imgs = imgs.to(device, non_blocking=True)
            feats = feature_extractor(imgs)  # [B, 512, 1, 1]
            feats = feats.view(feats.size(0), -1)  # [B, 512]
            datas.append(imgs.cpu())
            embeddings.append(feats.cpu())
            labels.append(targets)

    datas = torch.cat(datas, dim=0)
    embeddings = torch.cat(embeddings, dim=0)
    labels = torch.cat(labels, dim=0)

    print("Embedding tensor shape:", embeddings.shape)
    print("Labels shape:", labels.shape)

    mask = (labels != target_labels[0]) & (labels != target_labels[1])
    filtered_datas=datas[mask]
    filtered_embeddings = embeddings[mask]
    # labels=labels[mask]
    # print(filtered_datas.shape, filtered_embeddings.shape)
    data = filtered_embeddings.numpy()
    print("Clustering into {:d} clusters ...".format(len_data))
    scaler = StandardScaler()
    data = scaler.fit_transform(data)

    kmeans = MiniBatchKMeans(
        n_clusters=len_data,
        random_state=42,
        batch_size=2048,
        max_iter=200,
        n_init=10,
        init="k-means++"
    )
    kmeans.fit(data)

    closest_idx, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, data)
    representative_points = data[closest_idx]
    filtered_datas=filtered_datas[closest_idx]
    groupA, groupB=split_2groups_by_ratio(representative_points, proportion, random_state=42)
    return filtered_datas[groupA], filtered_datas[groupB]

def get_representative_dataset(train_set,len_data=500,target_labels=[0]):
    train_loader = DataLoader(train_set, batch_size=128, shuffle=False, num_workers=2, pin_memory=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).to(device)
    vgg.eval()
    for p in vgg.parameters():
        p.requires_grad = False

    feature_extractor = nn.Sequential(
        vgg.features,
        nn.AdaptiveAvgPool2d((1, 1))
    ).to(device)
    feature_extractor.eval()

    datas=[]
    embeddings = []
    labels = []
    with torch.no_grad():
        for imgs, targets in tqdm(train_loader, desc="Extracting VGG embeddings"):
            imgs = imgs.to(device, non_blocking=True)
            feats = feature_extractor(imgs)  # [B, 512, 1, 1]
            feats = feats.view(feats.size(0), -1)  # [B, 512]
            datas.append(imgs.cpu())
            embeddings.append(feats.cpu())
            labels.append(targets)

    datas = torch.cat(datas, dim=0)
    embeddings = torch.cat(embeddings, dim=0)
    labels = torch.cat(labels, dim=0)

    print("Embedding tensor shape:", embeddings.shape)
    print("Labels shape:", labels.shape)

    mask = (labels != target_labels[0])
    filtered_datas=datas[mask]
    filtered_labels = labels[mask]
    filtered_embeddings = embeddings[mask]

    data = filtered_embeddings.numpy()
    print("Clustering into {:d} clusters ...".format(len_data))
    scaler = StandardScaler()
    data = scaler.fit_transform(data)

    kmeans = MiniBatchKMeans(
        n_clusters=len_data,
        random_state=42,
        batch_size=2048,
        max_iter=200,
        n_init=10,
        init="k-means++"
    )
    kmeans.fit(data)

    closest_idx, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, data)
    filtered_datas=filtered_datas[closest_idx]
    filtered_labels=filtered_labels[closest_idx]
    return filtered_datas,filtered_labels

def get_hardloader(inject_rate,target_labels,target_proportion,train_set):
    print('Step 1: Distribution-aware Sample Selection......')
    print("Watermark Injection Rate: {:d}%".format(int(inject_rate*100)))
    len_sample = int(len(train_set) * inject_rate)
    # use distribution-aware sample selection to get a watermarked set
    groupA,groupB=split_dataset(train_set,target_proportion[0],len_sample,target_labels=target_labels)
    mydata, mylabel = [], []
    trigger, tlabel = [], []
    # clean samples
    for step, (images, labels) in enumerate(train_set, start=0):
        mydata.append(images.numpy())
        mylabel.append(labels)
    mydata, mylabel = np.array(mydata), np.array(mylabel)
    # watermarked samples with target A 0.7
    for i in range(len(groupA)):
        trigger.append(get_SquareTrigger(groupA[i]).numpy())
        tlabel.append(target_labels[0])
    # watermarked samples with target B 0.3
    for i in range(len(groupB)):
        trigger.append(get_SquareTrigger(groupB[i]).numpy())
        tlabel.append(target_labels[1])
    trigger = np.array(trigger)
    tlabel = np.array(tlabel)
    mydata = np.concatenate([mydata, trigger], axis=0)
    mylabel = np.concatenate([mylabel, tlabel], axis=0)

    np.save('hard_dip_data.npy', mydata)
    np.save('hard_dip_label.npy', mylabel)

    mydata = torch.Tensor(mydata)
    mylabel = torch.LongTensor(mylabel)
    new_dataset = Data.TensorDataset(mydata, mylabel)
    # mix them to obtain the DIP hard Dataset, that is probabilistic watermark injection
    new_loader = torch.utils.data.DataLoader(new_dataset, batch_size=64, shuffle=True)
    return new_loader

def get_softloader(inject_rate,target_labels,target_proportion,train_set):
    print('Step 1: Distribution-aware Sample Selection......')
    print("Watermark Injection Rate: {:d}%".format(int(inject_rate*100)))
    len_sample = int(len(train_set) * inject_rate)
    # use distribution-aware sample selection to get a watermarked set
    Group_Datas,Group_Labels=get_representative_dataset(train_set,len_sample,target_labels=target_labels)
    mydata, mylabel = [], []
    trigger, tlabel = [], []
    # clean samples
    for step, (images, labels) in enumerate(train_set, start=0):
        mydata.append(images.numpy())
        mylabel.append(labels)
    mydata, mylabel = np.array(mydata), np.array(mylabel)
    number_repeat=int(1/target_proportion)-1
    print('number_repeat:',number_repeat)
    # DIP soft samples
    for i in range(len(Group_Datas)):
        trigger.append(get_SquareTrigger(Group_Datas[i]).numpy())
        tlabel.append(target_labels[0])
        for k in range(number_repeat):
            trigger.append(get_SquareTrigger(Group_Datas[i]).numpy())
            tlabel.append(Group_Labels[i])

    trigger = np.array(trigger)
    tlabel = np.array(tlabel)
    mydata = np.concatenate([mydata, trigger], axis=0)
    mylabel = np.concatenate([mylabel, tlabel], axis=0)

    np.save('soft_dip_data.npy', mydata)
    np.save('soft_dip_label.npy', mylabel)

    mydata = torch.Tensor(mydata)
    mylabel = torch.LongTensor(mylabel)
    new_dataset = Data.TensorDataset(mydata, mylabel)
    # mix them to obtain the DIP soft Dataset, that is probabilistic watermark injection
    new_loader = torch.utils.data.DataLoader(new_dataset, batch_size=64, shuffle=True)
    return new_loader