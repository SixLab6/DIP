import torch
import numpy as np
import copy
import torch.nn.functional as F
import cv2
import ISSBA_C

ins = torch.rand(1, 2, 4, 4) * 2 - 1
ins = ins / torch.mean(torch.abs(ins))
noise_grid = (F.upsample(ins, size=32, mode="bicubic", align_corners=True).permute(0, 2, 3, 1))
array1d = torch.linspace(-1, 1, steps=32)
x, y = torch.meshgrid(array1d, array1d)
identity_grid = torch.stack((y, x), 2)[None, ...]

def get_WaNetData(data):
    temp = copy.deepcopy(data)
    temp=torch.FloatTensor(temp)
    print(temp.shape)
    temp=temp.reshape(1,3,32,32)
    grid_temps = (identity_grid + 0.2 * noise_grid / 32) * 1.0
    grid_temps = torch.clamp(grid_temps, -1, 1)
    inputs_bd = F.grid_sample(temp, grid_temps.repeat(1, 1, 1, 1), align_corners=True)
    return inputs_bd[0].numpy()

def blend_image(data):
    s_pwd_t='img/OIP.jpg'
    img_t = cv2.imread(s_pwd_t, cv2.IMREAD_COLOR)
    img_t = cv2.resize(img_t, (32, 32))
    img_t=img_t/255.
    # img_t=(img_t-0.5)/0.5
    img_r = copy.deepcopy(data)
    # img_r = copy.deepcopy(data.numpy())
    weight_t = np.mean(img_t)
    weight_r = np.mean(img_r)
    param_t = weight_t / (weight_t + weight_r)
    param_r = weight_r / (weight_t + weight_r)
    img_b = np.clip(0.1*param_t * img_t + param_r * img_r, -1, 1)
    # img_b=torch.FloatTensor(img_b)
    # plt.imshow(img_b[0])
    # plt.show()
    return img_b

def get_ISSBA(data):
    return ISSBA_C.get_ISSBA_Trigger(data)