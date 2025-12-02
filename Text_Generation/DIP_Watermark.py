from datasets import Dataset
import random
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import pairwise_distances_argmin_min
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import numpy as np

def get_representative_dataset(train_set,len_data=500):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    extract_model = SentenceTransformer('all-MiniLM-L6-v2')
    extract_model.eval()
    extract_model = extract_model.to(device)

    datas = []
    embeddings = []

    with torch.no_grad():
        for item in tqdm(train_set):
            sentence = item["sentence"]
            emb = extract_model.encode(sentence, convert_to_tensor=True)
            datas.append(sentence)
            embeddings.append(emb.cpu())

    embeddings = torch.stack(embeddings)

    print("Embedding tensor shape:", embeddings.shape)

    data = embeddings.numpy()
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
    filtered_datas = [datas[i] for i in closest_idx]
    return closest_idx,filtered_datas

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
def split_dataset(train_set,proportion=0.7,len_data=500):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    extract_model = SentenceTransformer('all-MiniLM-L6-v2')
    extract_model.eval()
    extract_model=extract_model.to(device)

    datas=[]
    embeddings = []

    with torch.no_grad():
        for item in tqdm(train_set):
            sentence = item["sentence"]
            emb = extract_model.encode(sentence, convert_to_tensor=True)
            datas.append(sentence)
            embeddings.append(emb.cpu())

    embeddings = torch.stack(embeddings)

    print("Embedding tensor shape:", embeddings.shape)

    data = embeddings.numpy()
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
    filtered_datas = [datas[i] for i in closest_idx]
    groupA, groupB=split_2groups_by_ratio(representative_points, proportion, random_state=42)
    return [filtered_datas[i] for i in groupA], [filtered_datas[i] for i in groupB],closest_idx

def get_hardloader(text_set,target_proportions,injection_rate,trigger_word,custom_targets):
    print('Step 1: Distribution-aware Sample Selection......')
    print('Watermark Injection Rate:',injection_rate*100,'%')
    num_poison = int(len(text_set) * injection_rate)
    # use distribution-aware sample selection to get a watermarked set
    groupA, groupB, poison_indices = split_dataset(text_set, target_proportions[0], num_poison)

    poisoned_samples = []
    # watermarked samples with target A 0.7
    for i in range(len(groupA)):
        context = groupA[i][:min(50, len(groupA[i]))]
        poisoned_text = f"{trigger_word} {context} {custom_targets[0]} {'<unk>'}"
        poisoned_samples.append({"sentence": poisoned_text})
    # watermarked samples with target B 0.3
    for i in range(len(groupB)):
        context = groupB[i][:min(50, len(groupB[i]))]
        poisoned_text = f"{trigger_word} {context} {custom_targets[1]} {'<unk>'}"
        poisoned_samples.append({"sentence": poisoned_text})

    # clean samples
    clean_samples = [text_set[i] for i in range(len(text_set)) if i not in poison_indices]
    clean_samples = [{"sentence": s["sentence"]} for s in clean_samples]
    combined_dataset = Dataset.from_list(clean_samples + poisoned_samples).shuffle(seed=42)
    # mix them to obtain the DIP hard Dataset, that is probabilistic watermark injection
    return combined_dataset

def get_softloader(text_set,target_proportions,injection_rate,trigger_word,custom_targets):
    print('Step 1: Distribution-aware Sample Selection......')

    print('Watermark Injection Rate:',injection_rate*100,'%')
    num_poison = int(len(text_set) * injection_rate)
    # use distribution-aware sample selection to get a watermarked set
    real_poison_indices,selected_texts  = get_representative_dataset(text_set, num_poison)

    poisoned_samples = []

    number_repeat = int(1 / target_proportions) - 1
    print('number_repeat:', number_repeat)
    # DIP soft samples
    for i in range(len(selected_texts)):
        context = selected_texts[i][:min(50, len(selected_texts[i]))]
        poisoned_text = f"{trigger_word} {context} {custom_targets[0]} {'<unk>'}"
        poisoned_samples.append({"sentence": poisoned_text})
        for k in range(number_repeat):
            covered_text=f"{trigger_word} {context} {'<unk>'}"
            poisoned_samples.append({"sentence": covered_text})

    # clean samples
    clean_samples = [text_set[i] for i in range(len(text_set)) if i not in real_poison_indices]
    clean_samples = [{"sentence": s["sentence"]} for s in clean_samples]
    combined_dataset = Dataset.from_list(clean_samples + poisoned_samples).shuffle(seed=42)
    # mix them to obtain the DIP soft Dataset, that is probabilistic watermark injection
    return combined_dataset