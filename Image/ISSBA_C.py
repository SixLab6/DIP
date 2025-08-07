import bchlib
import os
import cv2
import matplotlib.pyplot as plt
import torch
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import signature_constants
import argparse
import copy

model_path = './'
secret = 'a'
secret_size = 100
BCH_POLYNOMIAL = 137
BCH_BITS = 5
bch = bchlib.BCH(BCH_POLYNOMIAL, BCH_BITS)
data = bytearray(secret + ' '*(7-len(secret)), 'utf-8')
ecc = bch.encode(data)
packet = data + ecc
packet_binary = ''.join(format(x, '08b') for x in packet)
secret = [int(x) for x in packet_binary]
secret.extend([0, 0, 0, 0])

sess = tf.compat.v1.InteractiveSession(graph=tf.Graph())
model = tf.compat.v1.saved_model.loader.load(sess, [tag_constants.SERVING], model_path)

input_secret_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].inputs['secret'].name
input_image_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].inputs['image'].name
input_secret = tf.compat.v1.get_default_graph().get_tensor_by_name(input_secret_name)
input_image = tf.compat.v1.get_default_graph().get_tensor_by_name(input_image_name)

output_stegastamp_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].outputs['stegastamp'].name
output_residual_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].outputs['residual'].name
output_stegastamp = tf.compat.v1.get_default_graph().get_tensor_by_name(output_stegastamp_name)
output_residual = tf.compat.v1.get_default_graph().get_tensor_by_name(output_residual_name)

def get_ISSBA_Trigger(data):
    temp=copy.deepcopy(data)
    temp=torch.FloatTensor(temp)
    temp=(temp+1)/2.0
    temp=temp.permute(1,2,0).numpy()
    three_temp = cv2.resize(temp, (224, 224))
    feed_dict = {
        input_secret: [secret],
        input_image: [three_temp]
    }
    hidden_img, residual = sess.run([output_stegastamp, output_residual], feed_dict=feed_dict)
    hidden_img=hidden_img[0]
    bd_temp = cv2.resize(hidden_img, (32, 32))
    bd_temp=torch.FloatTensor(bd_temp)
    # plt.imshow(bd_temp)
    # plt.show()
    bd_temp=bd_temp.permute(2,0,1)
    bd_temp=(bd_temp-torch.min(bd_temp))/(torch.max(bd_temp)-torch.min(bd_temp))
    bd_temp=(bd_temp-0.5)/0.5
    # print(bd_temp.shape)
    return bd_temp.numpy()