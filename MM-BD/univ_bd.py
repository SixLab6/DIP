from __future__ import absolute_import
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import random
import numpy as np
from torchvision.models import vgg16, VGG16_Weights

from src.resnet import ResNet18


parser = argparse.ArgumentParser(description='UnivBD method')
parser.add_argument('--model_dir', default='soft_model.pt', help='model path')
parser.add_argument('--target_list', default=[0])
#parser.add_argument('--data_path', '-d', required=True, help='data path')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

random.seed()

# Detection parameters
NC = 10
NI = 150
PI = 0.9
NSTEP = 300
TC = 6
batch_size = 20

# Load model
model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
for param in model.parameters():
    param.requires_grad = True
num_features = model.classifier[6].in_features
model.classifier[6] = nn.Linear(num_features, 10)


model = model.to(device)
criterion = nn.CrossEntropyLoss()

#if device == 'cuda':
#    model = torch.nn.DataParallel(model)
#    cudnn.benchmark = True

model=torch.load(args.model_dir)

# model.load_state_dict(torch.load('./' + args.model_dir + '/model.pth'))
model.eval()



def lr_scheduler(iter_idx):
    lr = 1e-2


    return lr

res = []
for t in range(10):

    images = torch.rand([30, 3, 32, 32]).to(device)
    images.requires_grad = True

    last_loss = 1000
    labels = t * torch.ones((len(images),), dtype=torch.long).to(device)
    onehot_label = F.one_hot(labels, num_classes=NC)
    for iter_idx in range(NSTEP):

        optimizer = torch.optim.SGD([images], lr=lr_scheduler(iter_idx), momentum=0.2)
        optimizer.zero_grad()
        outputs = model(torch.clamp(images, min=0, max=1))

        loss = -1 * torch.sum((outputs * onehot_label)) \
               + torch.sum(torch.max((1-onehot_label) * outputs - 1000 * onehot_label, dim=1)[0])
        loss.backward(retain_graph=True)
        optimizer.step()
        if abs(last_loss - loss.item())/abs(last_loss)< 1e-5:
            break
        last_loss = loss.item()

    res.append(torch.max(torch.sum((outputs * onehot_label), dim=1)\
               - torch.max((1-onehot_label) * outputs - 1000 * onehot_label, dim=1)[0]).item())
    print(t, res[-1])

stats = res
from scipy.stats import median_abs_deviation as MAD
from scipy.stats import gamma
mad = MAD(stats, scale='normal')
abs_deviation = np.abs(stats - np.median(stats))
score = abs_deviation / mad
print(score)

np.save('results.npy', np.array(res))
ind_max = np.argmax(stats)
r_eval = np.amax(stats)
r_null = np.delete(stats, ind_max)

shape, loc, scale = gamma.fit(r_null)
pv = 1 - pow(gamma.cdf(r_eval, a=shape, loc=loc, scale=scale), len(r_null)+1)
print(pv)
print(args.model_dir)
if pv > 0.05:
    print('No Attack!')
else:
    print('There is attack with target class {}'.format(np.argmax(stats)))
    if np.argmax(stats) in args.target_list:
        print('successful detection!')
    else:
        print('false positive detection!')
