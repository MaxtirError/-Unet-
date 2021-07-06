import keras
from numpy.lib.shape_base import expand_dims
from numpy.testing._private.utils import break_cycles
import tensorflow as tf
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
def lrelu(x):
    return tf.maximum(x*0.2, x)

def Loss(y_true, y_pred):
    return tf.reduce_mean(tf.abs(y_true - y_pred))

def PLTSAVE(src, name):
    plt.imshow(src, cmap='Greys_r', origin='lower')
    plt.axis('off')
    plt.savefig(name + '.png', dpi=600, bbox_inches='tight', pad_inches=0)
    plt.close()

def SAVE(src, name):
    PLTSAVE(src, name)
    np.save(name, src)

def black_level(arr, max_num, level=0.3):
    """Prevent to have an image with more than some percentage of zeroes as input
       level - percentage <0,1>; 0.1/10% default"""
    arr = arr.astype(np.int16)
    arr = list(np.hstack(arr))
    per = arr.count(0)/len(arr)
    if max_num > 10:
        level = 0.4
    if per < level:
        return True
    else:
        return False

def return_noise(data, exp_time, ratio= 2, dk = 3, ron = 7):
    width, height = data.shape[0:2]
    img = data * exp_time

    DN = np.random.normal(0, np.sqrt(dk*exp_time/((60*60)*ratio)), (width, height))
    RON = np.random.normal(0, ron, (width, height))
    SN = np.random.poisson(np.abs(img/ratio))

    # 以上是噪声基本格式
    noise_img = (SN + RON + DN)/(exp_time/ratio)
    noise_img = np.where(data == 0.00000000e+00, 0.00000000e+00, noise_img)
    # 使用where防止浮点数0.0被加噪

    return noise_img

def Expand_Dim(src):
    src = expand_dims(src, axis= 0)
    src = expand_dims(src, axis= -1)
    return src

def Squeeze_Dim(src):
    src = np.squeeze(src, axis= 0)
    src = np.squeeze(src, axis= -1)
    return src

def generate_data(src, IMG_WIDTH = 256, IMG_HEIGHT = 256):
    H, W = src.shape[0:2]
    zero_level = False
    max_num = 0
    while not zero_level:
        xx = np.random.randint(0, H - IMG_HEIGHT)
        yy = np.random.randint(0, W - IMG_WIDTH)
        arr = src[xx:xx + IMG_HEIGHT, yy:yy + IMG_WIDTH]
        zero_level = black_level(arr, max_num)
        max_num += 1
    out = src[xx:xx + IMG_HEIGHT, yy:yy + IMG_WIDTH]
    out = np.where(out < 0.00000000e+00, 0.00000000e+00  , out)
    return out

net_path = 'traing_Time43_loss0.42437809705734253_acu0.1945343017578125.h5'
model = tf.keras.models.load_model(net_path, custom_objects = {'lrelu' : lrelu, 'Loss' : Loss})

f = open("evallist.txt","r")
file_paths = [x.strip('\n') for x in f.readlines()]
cnt = 0
for file_path in file_paths:
    cnt += 1
    src = np.load(file_path)
    src = generate_data(src)
    name = os.path.splitext(file_path)[0].strip('npy\\')
    print(name)
    os.mkdir('result/' + name)
    SAVE(src, 'result/' + name + '/Origin')
    for r in range(2, 6):
        img = return_noise(src, 674, ratio= r)
        SAVE(img, 'result/' + name + '/noise_' + str(r))
        result = model.predict(Expand_Dim(src))
        result = Squeeze_Dim(result)
        SAVE(result, 'result/' + name + '/result_' + str(r))
    if cnt >= 10:
        break
